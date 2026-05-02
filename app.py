import os
import re
import json
import uuid
import time
import shutil
import base64
import tempfile
import subprocess
import requests
from pathlib import Path

import streamlit as st
from PIL import Image, ImageDraw, ImageFont
from openai import OpenAI

# =========================
# 1. 基础配置与初始化
# =========================
st.set_page_config(page_title="AI 智能融合广告生成器", page_icon="🎬", layout="wide")
st.title("🎬 豆包视觉 × DeepSeek × Seedance：AI 智能广告融合")
st.caption("全链路：抽帧分析(豆包视觉) -> 导演决策与分镜(DeepSeek) -> 物理级视频渲染(豆包Seedance) -> 无缝合成(FFmpeg)")

OUTPUT_ROOT = Path.cwd() / "outputs"
OUTPUT_ROOT.mkdir(exist_ok=True)

# =========================
# 2. 核心工具与视频处理函数
# =========================
def find_executable(custom_value: str, fallback_name: str) -> str | None:
    return custom_value.strip() if custom_value.strip() else shutil.which(fallback_name)

def run_cmd(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"命令执行失败：{' '.join(cmd)}\nstderr:\n{result.stderr}")

def get_media_meta(ffprobe_bin: str, input_path: str) -> dict:
    cmd = [ffprobe_bin, "-v", "error", "-print_format", "json", "-show_format", "-show_streams", input_path]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0: raise RuntimeError(f"ffprobe 失败：{result.stderr}")
    info = json.loads(result.stdout)
    
    dur, w, h, has_audio = 0.0, 1280, 720, False
    try: dur = float(info.get("format", {}).get("duration", 0.0))
    except: pass
    for stream in info.get("streams", []):
        if stream.get("codec_type") == "video":
            w = int(stream.get("width") or w)
            h = int(stream.get("height") or h)
        if stream.get("codec_type") == "audio":
            has_audio = True
    return {"duration": dur, "width": w, "height": h, "has_audio": has_audio}

def extract_json_object(text: str) -> dict:
    match = re.search(r"\{.*\}", text, re.S)
    if match:
        try: return json.loads(match.group(0))
        except: pass
    return {}

def extract_keyframe(ffmpeg_bin: str, video_path: str, time_sec: float, out_img: str):
    """从指定秒数提取一帧作为视觉分析依据"""
    cmd = [ffmpeg_bin, "-y", "-ss", str(time_sec), "-i", video_path, "-vframes", "1", "-q:v", "2", out_img]
    subprocess.run(cmd, capture_output=True)

# =========================
# 3. AI 大模型 API 调用函数
# =========================
def analyze_image_with_doubao(image_path: str, prompt: str, api_key: str) -> str:
    """调用火山豆包多模态 API 分析画面内容"""
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    payload = {
        "model": "doubao-seed-2-0-pro-260215",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_base64}"}}
                ]
            }
        ]
    }
    
    resp = requests.post(url, headers=headers, json=payload)
    if resp.status_code == 200:
        data = resp.json()
        try:
            if "choices" in data and isinstance(data["choices"], list):
                return data["choices"][0]["message"]["content"]
            elif "output" in data and isinstance(data["output"], dict):
                return data["output"].get("choices", [{}])[0].get("message", {}).get("content", str(data))
            else:
                return f"未知的返回结构: {str(data)[:200]}"
        except Exception as e:
            return f"JSON解析错误: {e}"
    else:
        raise RuntimeError(f"豆包 API 请求失败 (状态码 {resp.status_code}): {resp.text}")

def plan_with_deepseek(api_key: str, model: str, orig_desc: str, ad_req: str, ad_type: str, dur: float, vision_contexts: str) -> dict:
    """调用 DeepSeek 结合视觉分析结果做出最终决策，并编写视频生成提示词"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"""
你是一个金牌广告剪辑导演。现在你需要为一个视频安排广告植入。
原视频总时长：{dur:.1f} 秒。
原视频基本简介：{orig_desc}
广告要求：{ad_req}
广告素材类型：{ad_type}

【关键画面视觉分析报告（由豆包AI视觉引擎提供）】
{vision_contexts}

请根据以上画面分析报告，选择一个**最自然的插入时间点**。
必须返回纯 JSON 格式：
{{
  "insert_time_sec": 插入的精确秒数 (必须在 1.0 到 {dur-1.0:.1f} 之间),
  "reason": "结合豆包的视觉描述，解释为什么选这个时间点最自然？",
  "method": "转场过渡 / 剧情插播",
  "video_prompt": "用一段极具画面感的提示词描述你要生成的广告视频（例如：特写镜头，一瓶清凉的饮料，水珠滑落，背景虚化，高画质电影感，4k，真实物理动态）。必须详细且有画面感！",
  "ad_script": "结合刚刚的画面内容，写一句承上启下的旁白台词（50字内）"
}}"""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return extract_json_object(resp.choices[0].message.content)

def generate_video_with_seedance(prompt: str, image_path: str, api_key: str, output_path: str) -> str:
    """调用火山豆包 Seedance 模型生成真实视频 (支持文生视频 / 图生视频)"""
    url_create = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    # 构造任务请求
    full_prompt = f"{prompt} --resolution 1080p --duration 5 --camerafixed false"
    content_list = [{"type": "text", "text": full_prompt}]

    # 如果有图片，转为【图生视频】
    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        ext = "jpeg" if image_path.lower().endswith("jpg") else "png"
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{img_base64}"}
        })

    payload = {
        "model": "doubao-seedance-1-0-pro-fast-251015",
        "content": content_list
    }

    resp = requests.post(url_create, headers=headers, json=payload)
    if resp.status_code != 200:
        raise RuntimeError(f"视频生成任务创建失败: {resp.text}")

    task_id = resp.json().get("id")
    if not task_id:
        raise RuntimeError(f"未获取到 Task ID: {resp.text}")

    # 轮询等待视频生成完成
    url_query = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
    
    with st.spinner("🎥 豆包 Seedance 正在渲染物理级真实视频 (约需1-2分钟，请耐心等待)..."):
        video_url = ""
        while True:
            q_resp = requests.get(url_query, headers=headers)
            if q_resp.status_code == 200:
                q_data = q_resp.json()
                status = q_data.get("status", "").lower()

                if status == "succeeded":
                    try:
                        video_url = q_data["content"]["video_url"]
                    except:
                        video_url = str(q_data)
                    break
                elif status in ["failed", "error"]:
                    raise RuntimeError(f"视频生成失败: {q_data}")
            
            time.sleep(5)

    if not video_url.startswith("http"):
        raise RuntimeError(f"未能解析到合法的视频地址，接口返回：{video_url}")

    # 下载视频
    video_data = requests.get(video_url).content
    with open(output_path, "wb") as f:
        f.write(video_data)
        
    return output_path

# =========================
# 4. FFmpeg 视频合成逻辑
# =========================
def build_final_video(ffmpeg, ffprobe, orig_path, ad_type, ad_input, ad_dur, plan, work_dir):
    meta = get_media_meta(ffprobe, orig_path)
    W, H, orig_dur = meta["width"], meta["height"], meta["duration"]
    insert_t = max(1.0, min(float(plan.get("insert_time_sec", orig_dur / 2)), orig_dur - 1.0))
    
    p1, ad_mp4, p2, concat_txt, final_out = work_dir/"p1.mp4", work_dir/"ad.mp4", work_dir/"p2.mp4", work_dir/"list.txt", work_dir/"final.mp4"
    
    norm_vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"
    base_cmd = ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]

    for seg, start, end in [(p1, 0, insert_t), (p2, insert_t, None)]:
        if start is not None and start >= orig_dur - 0.5: continue
        cmd = [ffmpeg, "-y"]
        if start: cmd += ["-ss", str(start)]
        if end: cmd += ["-to", str(end)]
        cmd += ["-i", orig_path]
        if not meta["has_audio"]: cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest", "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-vf", norm_vf] + base_cmd + [str(seg)]
        run_cmd(cmd)

    # 制作广告片段（此时 ad_input 必然已经是 MP4 视频了）
    ad_real_dur = get_media_meta(ffprobe, ad_input)["duration"] or ad_dur
    vf = f"{norm_vf},fade=t=in:st=0:d=0.5,fade=t=out:st={ad_real_dur-0.5}:d=0.5"
    cmd = [ffmpeg, "-y", "-i", ad_input, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest", "-map", "0:v:0", "-map", "1:a:0", "-vf", vf] + base_cmd + [str(ad_mp4)]
    run_cmd(cmd)

    clips = [p for p in [p1, ad_mp4, p2] if p.exists()]
    with open(concat_txt, "w") as f:
        for p in clips: f.write(f"file '{p.absolute().as_posix()}'\n")
    
    run_cmd([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", "-movflags", "+faststart", str(final_out)])
    return str(final_out)

# =========================
# 5. Streamlit 前端交互
# =========================
with st.sidebar:
    st.header("⚙️ 核心引擎配置")
    
    volces_key = st.text_input("火山引擎 (豆包) API Key", value="ark-20123d3b-09ac-4ede-b7c5-19a4c53f3dbc-9e361", type="password", help="用于豆包视觉分析 和 Seedance 视频生成")
    
    deepseek_key = st.text_input("DeepSeek API Key", value=os.getenv("DEEPSEEK_API_KEY", ""), type="password")
    deepseek_model = st.selectbox(
        "DeepSeek 模型版本", 
        ["deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro", "deepseek-reasoner"],
        index=0
    )
    
    ffmpeg_bin = st.text_input("ffmpeg 路径", "ffmpeg")
    ffprobe_bin = st.text_input("ffprobe 路径", "ffprobe")

col1, col2 = st.columns(2)
with col1:
    orig_desc = st.text_area("原视频大意", "一支Vlog视频")
    ad_req = st.text_area("广告核心要求", "植入一款饮料广告，要自然承接。")
    orig_vid = st.file_uploader("上传原视频", type=["mp4"])
with col2:
    ad_type = st.radio("广告形式", ["text", "image", "video"], horizontal=True)
    ad_dur = 5.0 # AI 视频生成固定时长通常为5秒左右
    
    ad_file, ad_text = None, ""
    if ad_type in ["image", "video"]: 
        ad_file = st.file_uploader(f"上传广告{ad_type}", type=["mp4", "png", "jpg", "jpeg"])
    else: 
        ad_text = st.text_area("广告文案", "补充能量，即刻出发！")

if st.button("🚀 开始 AI 智能视觉打点与视频融合", type="primary", use_container_width=True):
    if not (volces_key and deepseek_key and orig_vid):
        st.error("请填全 API Keys 并上传视频！")
        st.stop()
        
    job_dir = OUTPUT_ROOT / f"job_{uuid.uuid4().hex[:6]}"
    job_dir.mkdir(parents=True)
    orig_path = job_dir / "orig.mp4"
    orig_path.write_bytes(orig_vid.read())
    
    if ad_type in ["image", "video"] and ad_file:
        ad_input = job_dir / f"ad_{ad_file.name}"
        ad_input.write_bytes(ad_file.read())
        ad_input = str(ad_input)
    else: 
        ad_input = ad_text
    
    ffmpeg_exec, ffprobe_exec = find_executable(ffmpeg_bin, "ffmpeg"), find_executable(ffprobe_bin, "ffprobe")
    meta = get_media_meta(ffprobe_exec, str(orig_path))
    dur = meta["duration"]

    # ====== 阶段 1：豆包视觉抽帧分析 ======
    st.markdown("### 👁️ 豆包 AI 正在进行视觉打点分析...")
    vision_reports = ""
    cols = st.columns(3)
    
    for i, pct in enumerate([0.2, 0.5, 0.8]):
        t = dur * pct
        img_path = str(job_dir / f"frame_{i}.jpg")
        extract_keyframe(ffmpeg_exec, str(orig_path), t, img_path)
        
        with cols[i]:
            st.image(img_path, caption=f"第 {t:.1f} 秒截图")
            with st.spinner("豆包分析中..."):
                prompt = f"我要在这个视频里植入广告。要求是：{ad_req}。请描述画面的具体场景和人物动作，并分析这个瞬间是否适合作为广告切入点？"
                try:
                    analysis = analyze_image_with_doubao(img_path, prompt, volces_key)
                    st.success("分析完毕")
                    st.caption(analysis[:100] + "...") 
                    vision_reports += f"【时间点：{t:.1f}秒】\n画面描述：{analysis}\n\n"
                except Exception as e:
                    st.error(f"视觉分析失败: {e}")

    # ====== 阶段 2：DeepSeek 综合决策 ======
    st.markdown("### 🧠 DeepSeek 正在基于视觉报告进行导演决策...")
    with st.spinner(f"调用 {deepseek_model} ..."):
        try:
            plan = plan_with_deepseek(deepseek_key, deepseek_model, orig_desc, ad_req, ad_type, dur, vision_reports)
            st.json(plan)
        except Exception as e:
            st.error(f"DeepSeek 决策失败: {e}")
            st.stop()

    # ====== 阶段 3：豆包 Seedance 真实视频生成 ======
    if ad_type in ["text", "image"]:
        st.markdown("### 🎥 豆包 Seedance 正在无中生有渲染真实广告片段...")
        video_prompt = plan.get("video_prompt", f"生成一段高品质的视频，内容：{ad_text}")
        st.info(f"💡 DeepSeek 编写的视频分镜提示词：\n{video_prompt}")
        
        generated_ad_mp4 = str(job_dir / "seedance_generated.mp4")
        img_path_for_gen = ad_input if ad_type == "image" else None
        
        try:
            generate_video_with_seedance(
                prompt=video_prompt, 
                image_path=img_path_for_gen, 
                api_key=volces_key, 
                output_path=generated_ad_mp4
            )
            st.success("✅ AI 视频渲染完成！")
            
            # 【关键魔法】将类型转为视频，后续交给FFmpeg做标准视频拼接
            ad_type = "video"
            ad_input = generated_ad_mp4
            
        except Exception as e:
            st.error("视频生成失败")
            st.exception(e)
            st.stop()

    # ====== 阶段 4：FFmpeg 动态合成 ======
    st.markdown("### 🎬 FFmpeg 正在进行无缝拼接...")
    with st.spinner("视频合成中，请稍候..."):
        try:
            final_mp4 = build_final_video(ffmpeg_exec, ffprobe_exec, str(orig_path), ad_type, ad_input, ad_dur, plan, job_dir)
            st.success("🎉 融合完毕！")
            st.video(final_mp4)
            st.download_button("📥 下载带广告的成品视频", open(final_mp4, "rb"), "ai_ad_fused.mp4", "video/mp4", use_container_width=True)
        except Exception as e:
            st.error("视频处理出错！")
            st.exception(e)