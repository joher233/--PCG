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
st.title("🎬 豆包视觉 × DeepSeek × Seedance：无缝延展广告")
st.caption("全链路：抽帧分析 -> 导演决策 -> 提取插入点首帧 -> 视频无缝延展渲染 -> FFmpeg合成")

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
    """从指定秒数提取一帧"""
    cmd = [ffmpeg_bin, "-y", "-ss", str(time_sec), "-i", video_path, "-vframes", "1", "-q:v", "2", out_img]
    subprocess.run(cmd, capture_output=True)

# =========================
# 3. AI 大模型 API 调用函数
# =========================
def analyze_image_with_doubao(image_path: str, prompt: str, api_key: str) -> str:
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

def plan_with_deepseek(api_key: str, model: str, orig_desc: str, ad_req: str, dur: float, vision_contexts: str) -> dict:
    """高度定制化的无缝延展 Prompt"""
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"""
你是一名好莱坞级商业广告导演。我们要使用“图生视频延展技术”将广告无缝植入原视频。

【基础信息】
🎬 原视频总时长：{dur:.1f} 秒
📜 原视频大意：{orig_desc}
🎯 广告核心诉求：{ad_req}

【视觉侦察报告】
{vision_contexts}

【核心机制要求，非常重要！】
1. 你选定的 `insert_time_sec` 的画面，将被我们截取下来，作为AI视频生成的**【起始第一帧】**。
2. 你的 `video_prompt` 必须描述：**画面从这一帧开始，里面的人物/场景如何产生自然连续的动作，并巧妙过渡到广告产品的展示上。**（例如：画面中的主角手顺势一抬，手里变出了一瓶清凉的饮料，镜头推进给饮料特写，水珠滑落...）

请严格输出 JSON：
{{
  "insert_time_sec": 12.5,
  "reason": "为何选这个时间点的画面作为延展的起点？",
  "video_prompt": "基于选定时间的画面，描述接下来5秒的连续动态变化，以及产品的魔法般出现或展示过程。必须有极强画面感、动作连续性和高质量提示词(如4k, 电影感)",
  "ad_script": "配音台词（50字内）"
}}"""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return extract_json_object(resp.choices[0].message.content)

def generate_video_with_seedance(prompt: str, image_path: str, api_key: str, output_path: str) -> str:
    url_create = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }

    full_prompt = f"{prompt} --resolution 1080p --duration 5 --camerafixed false"
    content_list = [{"type": "text", "text": full_prompt}]

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
    
    url_query = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
    with st.spinner("🎥 豆包 Seedance 正在基于原视频画面进行物理延展渲染 (约需1-2分钟)..."):
        video_url = ""
        while True:
            q_resp = requests.get(url_query, headers=headers)
            if q_resp.status_code == 200:
                q_data = q_resp.json()
                status = q_data.get("status", "").lower()
                if status == "succeeded":
                    try: video_url = q_data["content"]["video_url"]
                    except: video_url = str(q_data)
                    break
                elif status in ["failed", "error"]:
                    raise RuntimeError(f"视频生成失败: {q_data}")
            time.sleep(5)

    video_data = requests.get(video_url).content
    with open(output_path, "wb") as f:
        f.write(video_data)
    return output_path

# =========================
# 4. FFmpeg 视频合成逻辑
# =========================
def build_final_video(ffmpeg, ffprobe, orig_path, ad_mp4_path, plan, work_dir):
    meta = get_media_meta(ffprobe, orig_path)
    W, H, orig_dur = meta["width"], meta["height"], meta["duration"]
    insert_t = max(1.0, min(float(plan.get("insert_time_sec", orig_dur / 2)), orig_dur - 1.0))
    
    p1, p2, concat_txt, final_out = work_dir/"p1.mp4", work_dir/"p2.mp4", work_dir/"list.txt", work_dir/"final.mp4"
    
    norm_vf = f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2,fps=30,setsar=1"
    base_cmd = ["-c:v", "libx264", "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "128k", "-ar", "44100", "-ac", "2"]

    # 切割原视频
    for seg, start, end in [(p1, 0, insert_t), (p2, insert_t, None)]:
        if start is not None and start >= orig_dur - 0.5: continue
        cmd = [ffmpeg, "-y"]
        if start: cmd += ["-ss", str(start)]
        if end: cmd += ["-to", str(end)]
        cmd += ["-i", orig_path]
        if not meta["has_audio"]: cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest", "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-vf", norm_vf] + base_cmd + [str(seg)]
        run_cmd(cmd)

    # 规范化生成的广告视频，仅做轻微音频淡入淡出（画面已是无缝所以不需要画面渐变）
    ad_norm = work_dir/"ad_norm.mp4"
    ad_dur = get_media_meta(ffprobe, ad_mp4_path)["duration"] or 5.0
    vf_ad = f"{norm_vf}" # 去掉了 fade 滤镜，因为我们要它和前一帧完全衔接！
    cmd = [ffmpeg, "-y", "-i", ad_mp4_path, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest", "-map", "0:v:0", "-map", "1:a:0", "-vf", vf_ad] + base_cmd + [str(ad_norm)]
    run_cmd(cmd)

    # 拼接
    clips = [p for p in [p1, ad_norm, p2] if p.exists()]
    with open(concat_txt, "w") as f:
        for p in clips: f.write(f"file '{p.absolute().as_posix()}'\n")
    
    run_cmd([ffmpeg, "-y", "-f", "concat", "-safe", "0", "-i", str(concat_txt), "-c", "copy", "-movflags", "+faststart", str(final_out)])
    return str(final_out)

# =========================
# 5. Streamlit 前端交互
# =========================
with st.sidebar:
    st.header("⚙️ 核心引擎配置")
    volces_key = st.text_input("火山引擎 (豆包) API Key", value="ark-20123d3b-09ac-4ede-b7c5-19a4c53f3dbc-9e361", type="password")
    deepseek_key = st.text_input("DeepSeek API Key", value="sk-ab72dc6429334244a7dc7f43e00e504c", type="password")
    deepseek_model = st.selectbox("DeepSeek 模型", ["deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro"], index=0)
    ffmpeg_bin = st.text_input("ffmpeg 路径", "ffmpeg")
    ffprobe_bin = st.text_input("ffprobe 路径", "ffprobe")

col1, col2 = st.columns(2)
with col1:
    orig_desc = st.text_area("原视频大意", "一支Vlog视频")
    orig_vid = st.file_uploader("上传原视频", type=["mp4"])
with col2:
    ad_req = st.text_area("广告核心要求", "广告内容：在这个场景中顺理成章地拿出一瓶运动饮料展示。")

if st.button("🚀 开始生成无缝延展广告", type="primary", use_container_width=True):
    if not (volces_key and deepseek_key and orig_vid):
        st.error("请填全 API Keys 并上传视频！")
        st.stop()
        
    job_dir = OUTPUT_ROOT / f"job_{uuid.uuid4().hex[:6]}"
    job_dir.mkdir(parents=True)
    orig_path = job_dir / "orig.mp4"
    orig_path.write_bytes(orig_vid.read())
    
    ffmpeg_exec, ffprobe_exec = find_executable(ffmpeg_bin, "ffmpeg"), find_executable(ffprobe_bin, "ffprobe")
    dur = get_media_meta(ffprobe_exec, str(orig_path))["duration"]

    # ====== 1. 抽帧分析 ======
    st.markdown("### 👁️ 豆包 AI 视觉打点分析...")
    vision_reports = ""
    cols = st.columns(3)
    for i, pct in enumerate([0.2, 0.5, 0.8]):
        t = dur * pct
        img_path = str(job_dir / f"frame_{i}.jpg")
        extract_keyframe(ffmpeg_exec, str(orig_path), t, img_path)
        with cols[i]:
            st.image(img_path, caption=f"第 {t:.1f} 秒")
            analysis = analyze_image_with_doubao(img_path, f"描述画面场景、人物动作，分析此处是否适合做广告插入转折点？广告要求：{ad_req}", volces_key)
            vision_reports += f"【{t:.1f}秒】画面：{analysis}\n"

    # ====== 2. DeepSeek 决策 ======
    st.markdown("### 🧠 DeepSeek 导演决策...")
    plan = plan_with_deepseek(deepseek_key, deepseek_model, orig_desc, ad_req, dur, vision_reports)
    st.json(plan)
    insert_t = max(1.0, min(float(plan.get("insert_time_sec", dur/2)), dur - 1.0))

    # ====== 3. 截取延展首帧并生成视频 ======
    st.markdown("### 🎥 提取过渡帧并进行 AI 物理延展生成...")
    transition_frame = str(job_dir / "transition_frame.jpg")
    extract_keyframe(ffmpeg_exec, str(orig_path), insert_t, transition_frame)
    st.image(transition_frame, caption=f"🔗 提取第 {insert_t:.1f} 秒作为 AI 生成起始帧 (无缝衔接保证)")
    
    generated_ad_mp4 = str(job_dir / "seedance_generated.mp4")
    generate_video_with_seedance(plan.get("video_prompt", ""), transition_frame, volces_key, generated_ad_mp4)
    st.success("✅ 视频延展生成完毕！")

    # ====== 4. 合成 ======
    st.markdown("### 🎬 FFmpeg 无缝合成拼接...")
    final_mp4 = build_final_video(ffmpeg_exec, ffprobe_exec, str(orig_path), generated_ad_mp4, plan, job_dir)
    st.video(final_mp4)