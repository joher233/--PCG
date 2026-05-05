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
st.title("🎬 豆包视觉 × DeepSeek × Seedance 智能广告融合")
st.caption("全链路：双向视觉分析(原片+素材) -> 导演决策(基于问卷) -> AI渲染/延展(带降级策略) -> 无缝合成")

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
def analyze_image_with_doubao(image_path: str, prompt: str, api_keys: list) -> str:
    with open(image_path, "rb") as f:
        img_base64 = base64.b64encode(f.read()).decode("utf-8")
    
    url = "https://ark.cn-beijing.volces.com/api/v3/chat/completions"
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
    
    last_err = ""
    for api_key in api_keys:
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
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
            last_err = f"HTTP {resp.status_code}: {resp.text}"
            continue 
            
    raise RuntimeError(f"豆包视觉 API 请求全部失败，最后错误: {last_err}")

def plan_with_deepseek(api_key: str, model: str, orig_desc: str, dur: float, vision_contexts: str, ad_type: str, ad_material_desc: str) -> dict:
    client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")
    prompt = f"""
作为一名顶尖的新媒体商业广告导演，你要将广告无缝植入原视频。

【受众调研铁律】
1. 绝对禁忌：91.3%的观众反感“正片中间突然打断”！你的植入点绝不能在动作进行一半时切断。
2. 最佳位置：优先考虑片尾结束处、片头，或“不影响内容的绝对自然过渡点”（如场景切换、黑场、情绪留白处）。
3. 创意为王：植入必须像原生内容的一部分。

【原视频情报】
🎬 原视频总时长：{dur:.1f} 秒
📜 原视频大意：{orig_desc}
👁️ 原视频关键帧视觉分析：
{vision_contexts}

【广告素材情报（非常重要）】
📦 客户提供的素材格式：{ad_type}
🔍 广告素材内容的 AI 视觉解析：
{ad_material_desc}

【任务要求】
请结合“原视频画面”和“广告素材特征”，决定最完美的植入时间和方式。
严格输出 JSON（不要输出多余解释）：
{{
  "insert_time_sec": 插入的精确秒数 (必须在 1.0 到 {dur-1.0:.1f} 之间),
  "reason": "深度解析：为什么在这个点插入这个广告素材最自然、最不容易被观众讨厌？",
  "video_prompt": "【仅当素材为文字/图片时有效】请写给 Seedance 视频大模型的高级提示词。要求：描述连续的物理动作、有创意、高画质(如4k, 电影感)。",
  "ad_script": "配音台词（50字内，巧妙衔接原片与广告）"
}}"""
    
    resp = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        response_format={"type": "json_object"},
        temperature=0.7
    )
    return extract_json_object(resp.choices[0].message.content)

def generate_video_with_seedance(prompt: str, image_path: str, api_configs: list, output_path: str) -> str:
    """调用火山豆包 Seedance 模型生成真实视频 (动态匹配 Key 和 Model)"""
    url_create = "https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks"
    
    full_prompt = f"{prompt} --duration 5 --camerafixed false"
    content_list = [{"type": "text", "text": full_prompt}]

    if image_path and os.path.exists(image_path):
        with open(image_path, "rb") as f:
            img_base64 = base64.b64encode(f.read()).decode("utf-8")
        ext = "jpeg" if image_path.lower().endswith("jpg") else "png"
        content_list.append({
            "type": "image_url",
            "image_url": {"url": f"data:image/{ext};base64,{img_base64}"}
        })

    task_id = None
    working_key = None
    working_model = None
    last_err = ""
    
    # 遍历主备配置，根据对应的 Key 和 Model 发起请求
    for config in api_configs:
        api_key = config["key"]
        model_name = config["model"]
        
        payload = {
            "model": model_name,
            "content": content_list
        }
        
        headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
        resp = requests.post(url_create, headers=headers, json=payload)
        
        if resp.status_code == 200 and resp.json().get("id"):
            task_id = resp.json().get("id")
            working_key = api_key
            working_model = model_name
            break
        else:
            last_err = f"[{model_name}] HTTP {resp.status_code}: {resp.text}"
            
    if not task_id:
        raise RuntimeError(f"视频生成任务创建失败(主备配置均无效): {last_err}")

    # 使用创建成功的 Key 轮询等待视频生成完成
    poll_headers = {"Authorization": f"Bearer {working_key}", "Content-Type": "application/json"}
    url_query = f"https://ark.cn-beijing.volces.com/api/v3/contents/generations/tasks/{task_id}"
    
    with st.spinner(f"🎥 正在使用 {working_model} 渲染高级动态视频 (约需1-2分钟)..."):
        video_url = ""
        while True:
            q_resp = requests.get(url_query, headers=poll_headers)
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

    for seg, start, end in [(p1, 0, insert_t), (p2, insert_t, None)]:
        if start is not None and start >= orig_dur - 0.5: continue
        cmd = [ffmpeg, "-y"]
        if start: cmd += ["-ss", str(start)]
        if end: cmd += ["-to", str(end)]
        cmd += ["-i", orig_path]
        if not meta["has_audio"]: cmd += ["-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest", "-map", "0:v:0", "-map", "1:a:0"]
        cmd += ["-vf", norm_vf] + base_cmd + [str(seg)]
        run_cmd(cmd)

    ad_norm = work_dir/"ad_norm.mp4"
    cmd = [ffmpeg, "-y", "-i", ad_mp4_path, "-f", "lavfi", "-i", "anullsrc=channel_layout=stereo:sample_rate=44100", "-shortest", "-map", "0:v:0", "-map", "1:a:0", "-vf", norm_vf] + base_cmd + [str(ad_norm)]
    run_cmd(cmd)

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
    
    st.subheader("火山引擎 (豆包) API Keys")
    volces_key_primary = st.text_input("主 Key (1.5 Pro)", value="ark-55944f19-c838-49f2-971c-ca703b3980f1-f04e1", type="password")
    volces_key_backup1 = st.text_input("备用 Key 1 (1.0 Fast)", value="ark-55944f19-c838-49f2-971c-ca703b3980f1-f04e1", type="password")
    volces_key_backup2 = st.text_input("备用 Key 2 (1.0 Fast)", value="ark-ff09d587-8442-432e-91e0-88ba3886d634-86277", type="password")
    
    # 构造 主备 Key 和 对应模型的配置列表
    volces_configs = []
    if volces_key_primary.strip():
        volces_configs.append({"key": volces_key_primary.strip(), "model": "doubao-seedance-1-5-pro-251215"})
    if volces_key_backup1.strip():
        volces_configs.append({"key": volces_key_backup1.strip(), "model": "doubao-seedance-1-0-pro-fast-251015"})
    if volces_key_backup2.strip():
        volces_configs.append({"key": volces_key_backup2.strip(), "model": "doubao-seedance-1-0-pro-fast-251015"})
        
    # 单独提取 keys 用于视觉分析函数 (视觉分析模型统一固定)
    volces_keys_only = [c["key"] for c in volces_configs]
    
    st.subheader("DeepSeek 决策大脑")
    deepseek_key = st.text_input("DeepSeek API Key", value="sk-be2f2294c2f34707a6452855a7441c76", type="password")
    deepseek_model = st.selectbox("DeepSeek 模型", ["deepseek-chat", "deepseek-v4-flash", "deepseek-v4-pro"], index=0)
    
    st.subheader("本地工具")
    ffmpeg_bin = st.text_input("ffmpeg 路径", "ffmpeg")
    ffprobe_bin = st.text_input("ffprobe 路径", "ffprobe")

col1, col2 = st.columns(2)
with col1:
    orig_desc = st.text_area("原视频大意", "一支日常Vlog视频")
    orig_vid = st.file_uploader("上传原视频", type=["mp4"])
with col2:
    ad_type = st.radio("广告素材类型", ["text", "image", "video"], horizontal=True)
    
    ad_file = None
    ad_text_req = ""
    
    if ad_type == "text":
        ad_text_req = st.text_area("输入广告诉求/文案", "补充能量，即刻出发的饮料广告！")
    elif ad_type == "image":
        ad_file = st.file_uploader("上传广告产品图片", type=["png", "jpg", "jpeg"])
    elif ad_type == "video":
        ad_file = st.file_uploader("上传成片广告视频", type=["mp4"])

if st.button("🚀 开始双向分析与智能融合", type="primary", use_container_width=True):
    if not (volces_configs and deepseek_key and orig_vid):
        st.error("请填全 API Keys 并上传原视频！")
        st.stop()
    if ad_type in ["image", "video"] and not ad_file:
        st.error(f"请上传广告 {ad_type} 素材！")
        st.stop()
        
    job_dir = OUTPUT_ROOT / f"job_{uuid.uuid4().hex[:6]}"
    job_dir.mkdir(parents=True)
    
    orig_path = job_dir / "orig.mp4"
    orig_path.write_bytes(orig_vid.read())
    
    ad_input_path = ""
    if ad_file:
        ad_input_path = str(job_dir / f"ad_material_{ad_file.name}")
        with open(ad_input_path, "wb") as f:
            f.write(ad_file.read())
    
    ffmpeg_exec, ffprobe_exec = find_executable(ffmpeg_bin, "ffmpeg"), find_executable(ffprobe_bin, "ffprobe")
    dur = get_media_meta(ffprobe_exec, str(orig_path))["duration"]

    # ====== 1. 原视频抽帧分析 ======
    st.markdown("### 👁️ 豆包 AI 分析原视频节奏...")
    vision_reports = ""
    cols = st.columns(3)
    for i, pct in enumerate([0.1, 0.5, 0.95]):
        t = dur * pct
        img_path = str(job_dir / f"orig_frame_{i}.jpg")
        extract_keyframe(ffmpeg_exec, str(orig_path), t, img_path)
        with cols[i]:
            st.image(img_path, caption=f"原片 第 {t:.1f} 秒")
            analysis = analyze_image_with_doubao(img_path, "描述画面场景、人物动作，分析此处是否适合做广告插入点？", volces_keys_only)
            vision_reports += f"【原片 {t:.1f}秒】画面：{analysis}\n"

    # ====== 2. 广告素材智能分析 ======
    st.markdown("### 🔍 豆包 AI 分析广告素材...")
    ad_material_desc = ""
    if ad_type == "text":
        ad_material_desc = f"纯文本诉求：{ad_text_req}"
        st.info(f"📝 {ad_material_desc}")
    elif ad_type == "image":
        with st.spinner("分析图片广告中..."):
            ad_material_desc = analyze_image_with_doubao(ad_input_path, "详细描述这张图片中的产品、品牌属性和核心视觉元素。", volces_keys_only)
        st.image(ad_input_path, width=200)
        st.success(f"🖼️ 素材解析：{ad_material_desc}")
    elif ad_type == "video":
        with st.spinner("抽取并分析视频广告关键帧..."):
            ad_frame_path = str(job_dir / "ad_video_frame.jpg")
            ad_dur = get_media_meta(ffprobe_exec, ad_input_path)["duration"]
            extract_keyframe(ffmpeg_exec, ad_input_path, ad_dur/2, ad_frame_path)
            ad_material_desc = analyze_image_with_doubao(ad_frame_path, "这是将要植入的广告视频中间帧，请描述其画面内容、产品特征及风格氛围。", volces_keys_only)
        st.image(ad_frame_path, width=200)
        st.success(f"🎬 视频解析：{ad_material_desc}")

    # ====== 3. DeepSeek 决策 ======
    st.markdown("### 🧠 DeepSeek 导演全局决策...")
    plan = plan_with_deepseek(deepseek_key, deepseek_model, orig_desc, dur, vision_reports, ad_type, ad_material_desc)
    st.json(plan)
    insert_t = max(1.0, min(float(plan.get("insert_time_sec", dur/2)), dur - 1.0))

    # ====== 4. 视频生成或获取 ======
    st.markdown("### ⚙️ 处理植入广告片段...")
    final_ad_video_path = ""
    
    if ad_type == "text":
        transition_frame = str(job_dir / "transition_frame.jpg")
        extract_keyframe(ffmpeg_exec, str(orig_path), insert_t, transition_frame)
        st.image(transition_frame, caption=f"提取原片第 {insert_t:.1f} 秒作为无缝延展起点")
        
        final_ad_video_path = str(job_dir / "seedance_generated.mp4")
        generate_video_with_seedance(plan.get("video_prompt", ""), transition_frame, volces_configs, final_ad_video_path)
        st.success("✅ 原视频首帧延展生成完毕！")
        
    elif ad_type == "image":
        final_ad_video_path = str(job_dir / "seedance_generated.mp4")
        generate_video_with_seedance(plan.get("video_prompt", ""), ad_input_path, volces_configs, final_ad_video_path)
        st.success("✅ 产品图片动效渲染完毕！")
        
    elif ad_type == "video":
        final_ad_video_path = ad_input_path
        st.success("✅ 直接采纳用户提供的视频素材进行智能点位植入。")

    # ====== 5. FFmpeg 合成 ======
    st.markdown("### 🎬 FFmpeg 最终合成...")
    final_mp4 = build_final_video(ffmpeg_exec, ffprobe_exec, str(orig_path), final_ad_video_path, plan, job_dir)
    st.video(final_mp4)
    st.balloons()