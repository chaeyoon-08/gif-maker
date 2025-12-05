import streamlit as st
import torch
from diffusers import StableVideoDiffusionPipeline
from diffusers.utils import load_image, export_to_gif
import tempfile
import os
import gc

st.set_page_config(page_title="GIF-Maker", page_icon="🎥", layout="centered")

st.title("🎥 GIF-Maker")
st.markdown("### 정지된 사진에 생명을 불어넣으세요 ✨")
st.markdown("Powered by **Stable Video Diffusion (SVD-XT)**")

if 'generated_gif' not in st.session_state:
    st.session_state.generated_gif = None

@st.cache_resource
def load_model():
    pipe = StableVideoDiffusionPipeline.from_pretrained(
        "stabilityai/stable-video-diffusion-img2vid-xt",
        torch_dtype=torch.float16,
        variant="fp16"
    )
    pipe.enable_model_cpu_offload()
    return pipe

with st.sidebar:
    st.header("⚙️ 설정")
    seed = st.number_input("Seed (고정 시드)", value=42)
    motion_bucket_id = st.slider("움직임 강도 (Motion Bucket)", 1, 255, 127)
    st.caption("값이 클수록 움직임이 커지지만, 이미지가 왜곡될 수 있습니다.")

uploaded_file = st.file_uploader("이미지를 업로드하세요 (JPG/PNG)", type=["jpg", "png", "jpeg"])

if uploaded_file:
    st.image(uploaded_file, caption="Original Image", use_container_width=True)
    
    if st.button("영상 생성 시작! 🚀", type="primary"):
        with st.spinner("🎬 영상을 생성하고 있습니다... (GPU 연산 중)"):
            try:
                torch.cuda.empty_cache()
                gc.collect()

                pipe = load_model()
                
                with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                    tmp.write(uploaded_file.getvalue())
                    tmp_path = tmp.name

                image = load_image(tmp_path)
                image = image.resize((1024, 576))

                generator = torch.manual_seed(seed)
                frames = pipe(
                    image, 
                    decode_chunk_size=2,
                    generator=generator,
                    motion_bucket_id=motion_bucket_id
                ).frames[0]

                output_path = "output.gif"
                export_to_gif(frames, output_path)

                st.session_state.generated_gif = output_path
                os.remove(tmp_path)
            
            except Exception as e:
                st.error(f"오류 발생: {e}")
                st.warning("⚠️ GPU 메모리가 부족할 수 있습니다.")

if st.session_state.generated_gif is not None:
    st.divider()
    st.success("🎉 영상 생성이 완료되었습니다!")
    st.image(st.session_state.generated_gif, caption="Generated GIF", use_container_width=True)
    
    with open(st.session_state.generated_gif, "rb") as f:
        file_data = f.read()
        st.download_button(
            label="💾 GIF 다운로드",
            data=file_data,
            file_name="gif_maker_result.gif",
            mime="image/gif"
        )