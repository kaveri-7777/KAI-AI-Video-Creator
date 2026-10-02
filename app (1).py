
import os
import re
import wave
import numpy as np
import torch
import gradio as gr

from PIL import Image, ImageDraw, ImageFont
from gtts import gTTS
from pydub import AudioSegment

from diffusers import StableDiffusionPipeline
from moviepy.editor import ImageClip, concatenate_videoclips
from moviepy.editor import VideoFileClip, AudioFileClip, CompositeVideoClip


# =========================================================
# KAI SETTINGS
# =========================================================

APP_NAME = "KAI"
OUTPUT_DIR = "KAI_Output"
IMAGE_DIR = os.path.join(OUTPUT_DIR, "images")
AUDIO_DIR = os.path.join(OUTPUT_DIR, "audio")
VIDEO_DIR = os.path.join(OUTPUT_DIR, "videos")
SUBTITLE_DIR = os.path.join(OUTPUT_DIR, "subtitles")

for folder in [
    OUTPUT_DIR,
    IMAGE_DIR,
    AUDIO_DIR,
    VIDEO_DIR,
    SUBTITLE_DIR
]:
    os.makedirs(folder, exist_ok=True)


# =========================================================
# LOAD AI IMAGE MODEL
# =========================================================

print("🎨 Loading KAI image generator...")

device = "cuda" if torch.cuda.is_available() else "cpu"

image_model = StableDiffusionPipeline.from_pretrained(
    "runwayml/stable-diffusion-v1-5",
    torch_dtype=torch.float16 if device == "cuda" else torch.float32
)

image_model = image_model.to(device)

print("✅ KAI image generator ready!")


# =========================================================
# SPLIT STORY
# =========================================================

def split_story(story):

    parts = re.split(
        r'[.!?。\n]+',
        story
    )

    parts = [
        p.strip()
        for p in parts
        if p.strip()
    ]

    if not parts:
        return []

    if len(parts) > 6:
        parts = parts[:6]

    while len(parts) < 6:
        parts.append(parts[-1])

    return parts[:6]


# =========================================================
# CREATE AI SCENES
# =========================================================

def create_scene_images(
    story,
    character_type,
    character_name,
    character_color,
    clothes,
    style
):

    scenes = split_story(story)

    style_prompts = {

        "Cute 3D Cartoon":
        """
        cute colorful 3D cartoon animation,
        big expressive eyes,
        soft rounded characters,
        bright children's colors,
        playful wholesome appearance
        """,

        "Storybook":
        """
        beautiful children's storybook illustration,
        soft hand-painted look,
        warm colors,
        magical storybook atmosphere
        """,

        "Colorful Animation":
        """
        bright colorful children's animation,
        vibrant environment,
        expressive characters,
        playful animated movie style
        """
    }

    selected_style = style_prompts.get(
        style,
        style_prompts["Cute 3D Cartoon"]
    )

    if not character_name.strip():
        character_name = "the main character"

    if not clothes.strip():
        clothes = "cute simple clothes"

    character_description = f"""
    Main character:
    {character_name}, a {character_color} {character_type}.
    Wearing {clothes}.
    Cute friendly face, expressive eyes,
    child-friendly proportions.
    Keep the SAME character design in every scene.
    """

    images = []

    for i, scene in enumerate(scenes, 1):

        print(f"🎬 Creating Scene {i}/6...")

        prompt = f"""
        {selected_style}

        {character_description}

        Vertical 9:16 composition.
        Wholesome children's content.
        Consistent character appearance.
        Bright colorful environment.

        Scene:
        {scene}
        """

        generator = torch.Generator(
            device=device
        ).manual_seed(9000 + i)

        image = image_model(
            prompt,
            height=768,
            width=432,
            num_inference_steps=20,
            guidance_scale=7.5,
            generator=generator
        ).images[0]

        image_path = os.path.join(
            IMAGE_DIR,
            f"scene_{i}.png"
        )

        image.save(image_path)

        images.append(image)

    return images, scenes


# =========================================================
# CREATE MOVING-STYLE VIDEO
# =========================================================

def create_base_video(images):

    clips = []

    for i, image in enumerate(images):

        array = np.array(
            image.convert("RGB")
        )

        clip = ImageClip(
            array
        ).set_duration(4)

        if i % 2 == 0:

            clip = clip.resize(
                lambda t:
                1 + 0.06 * t / 4
            )

        else:

            clip = clip.resize(
                lambda t:
                1.06 - 0.06 * t / 4
            )

        clips.append(clip)

    video = concatenate_videoclips(
        clips,
        method="compose"
    )

    path = os.path.join(
        VIDEO_DIR,
        "KAI_Base_Video.mp4"
    )

    video.write_videofile(
        path,
        fps=24,
        codec="libx264",
        logger=None
    )

    return path


# =========================================================
# CREATE CUTE TELUGU VOICE + MELODY
# =========================================================

def create_audio(
    scenes,
    voice_style
):

    settings = {

        "Cute Girl": 1.20,
        "Soft Girl": 1.12,
        "Playful": 1.25,
        "Normal": 1.00
    }

    pitch = settings.get(
        voice_style,
        1.20
    )

    voice_track = AudioSegment.empty()

    for i, text in enumerate(scenes):

        temp_file = os.path.join(
            AUDIO_DIR,
            f"voice_{i+1}.mp3"
        )

        tts = gTTS(
            text=text,
            lang="te",
            slow=False
        )

        tts.save(temp_file)

        part = AudioSegment.from_mp3(
            temp_file
        )

        new_rate = int(
            part.frame_rate * pitch
        )

        part = part._spawn(
            part.raw_data,
            overrides={
                "frame_rate": new_rate
            }
        ).set_frame_rate(
            part.frame_rate
        )

        if len(part) < 4000:

            part += AudioSegment.silent(
                duration=4000 - len(part)
            )

        else:

            part = part[:4000]

        voice_track += part

    # ---------------------------------------
    # Simple original melody
    # ---------------------------------------

    sample_rate = 44100
    total_seconds = len(scenes) * 4

    t = np.linspace(
        0,
        total_seconds,
        int(sample_rate * total_seconds),
        endpoint=False
    )

    notes = [
        261.63,
        329.63,
        392.00,
        329.63,
        293.66,
        349.23,
        440.00,
        349.23
    ]

    melody = np.zeros_like(t)

    for i in range(
        int(total_seconds / 0.5)
    ):

        freq = notes[
            i % len(notes)
        ]

        start = int(
            i * 0.5 * sample_rate
        )

        end = min(
            len(t),
            int(
                (i * 0.5 + 0.5)
                * sample_rate
            )
        )

        local_t = np.linspace(
            0,
            (end - start) / sample_rate,
            end - start,
            endpoint=False
        )

        melody[start:end] += (
            np.sin(
                2 * np.pi * freq * local_t
            ) * 0.08
        )

    melody = melody / max(
        np.max(np.abs(melody)),
        1
    )

    melody_int16 = (
        melody * 32767
    ).astype(np.int16)

    melody_file = os.path.join(
        AUDIO_DIR,
        "melody.wav"
    )

    with wave.open(
        melody_file,
        "w"
    ) as wav:

        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(sample_rate)

        wav.writeframes(
            melody_int16.tobytes()
        )

    music = AudioSegment.from_wav(
        melody_file
    )

    final_audio = music.overlay(
        voice_track - 2
    )

    output_audio = os.path.join(
        AUDIO_DIR,
        "KAI_Cute_Telugu_Song.mp3"
    )

    final_audio.export(
        output_audio,
        format="mp3"
    )

    return output_audio


# =========================================================
# CREATE TELUGU SUBTITLES
# =========================================================

def create_subtitles(
    video,
    scenes
):

    font_path = (
        "/usr/share/fonts/truetype/noto/"
        "NotoSansTelugu-Regular.ttf"
    )

    if not os.path.exists(font_path):

        os.system(
            "apt-get -qq update"
        )

        os.system(
            "apt-get -qq install fonts-noto-core"
        )

    font = ImageFont.truetype(
        font_path,
        25
    )

    clips = []

    duration = video.duration / len(scenes)

    for i, text in enumerate(scenes):

        img = Image.new(
            "RGBA",
            (432, 130),
            (0, 0, 0, 0)
        )

        draw = ImageDraw.Draw(img)

        draw.rounded_rectangle(
            (8, 5, 424, 122),
            radius=15,
            fill=(0, 0, 0, 165)
        )

        bbox = draw.multiline_textbbox(
            (0, 0),
            text,
            font=font,
            spacing=6
        )

        width = bbox[2] - bbox[0]

        x = max(
            10,
            (432 - width) // 2
        )

        draw.multiline_text(
            (x, 20),
            text,
            font=font,
            fill="white",
            stroke_width=2,
            stroke_fill="black",
            spacing=6,
            align="center"
        )

        subtitle_file = os.path.join(
            SUBTITLE_DIR,
            f"subtitle_{i+1}.png"
        )

        img.save(
            subtitle_file
        )

        clip = (
            ImageClip(subtitle_file)
            .set_start(i * duration)
            .set_duration(duration)
            .set_position(
                ("center", 595)
            )
        )

        clips.append(clip)

    return clips


# =========================================================
# COMPLETE KAI GENERATOR
# =========================================================

def generate_kai_video(
    story,
    character_type,
    character_name,
    character_color,
    clothes,
    style,
    voice_style,
    progress=gr.Progress()
):

    if not story.strip():

        raise gr.Error(
            "Please enter a story."
        )

    progress(
        0.05,
        desc="🤖 KAI is reading your story..."
    )

    progress(
        0.10,
        desc="🎨 Creating AI scenes..."
    )

    images, scenes = create_scene_images(
        story,
        character_type,
        character_name,
        character_color,
        clothes,
        style
    )

    progress(
        0.65,
        desc="🎬 Creating video..."
    )

    base_video_path = create_base_video(
        images
    )

    progress(
        0.75,
        desc="🎵 Creating Telugu voice and melody..."
    )

    audio_path = create_audio(
        scenes,
        voice_style
    )

    progress(
        0.85,
        desc="📝 Adding Telugu subtitles..."
    )

    video = VideoFileClip(
        base_video_path
    )

    audio = AudioFileClip(
        audio_path
    )

    if audio.duration < video.duration:

        from moviepy.audio.fx.all import audio_loop

        audio = audio_loop(
            audio,
            duration=video.duration
        )

    else:

        audio = audio.subclip(
            0,
            video.duration
        )

    video = video.set_audio(
        audio
    )

    subtitle_clips = create_subtitles(
        video,
        scenes
    )

    final_video = CompositeVideoClip(
        [video] + subtitle_clips
    )

    final_path = os.path.join(
        VIDEO_DIR,
        "KAI_Final_Video.mp4"
    )

    final_video.write_videofile(
        final_path,
        fps=24,
        codec="libx264",
        audio_codec="aac",
        logger=None
    )

    progress(
        1.0,
        desc="🎉 KAI video complete!"
    )

    return final_path


# =========================================================
# KAI APP
# =========================================================

with gr.Blocks(
    title="KAI - AI Kids Video Creator",
    theme=gr.themes.Soft()
) as app:

    gr.Markdown("""
    # 🤖 KAI
    ## AI Kids Story Video Creator 🎬

    **Story → Characters → AI Scenes → Voice → Song → Subtitles**
    """)

    story = gr.Textbox(
        label="📝 Your Story",
        placeholder="మీ పిల్లల కథను ఇక్కడ రాయండి...",
        lines=8
    )

    with gr.Row():

        character_type = gr.Dropdown(
            [
                "Rabbit",
                "Cat",
                "Dog",
                "Bear",
                "Girl"
            ],
            value="Rabbit",
            label="🐾 Character"
        )

        character_color = gr.Dropdown(
            [
                "White",
                "Brown",
                "Black",
                "Golden",
                "Pink",
                "Blue"
            ],
            value="White",
            label="🎨 Color"
        )

    with gr.Row():

        character_name = gr.Textbox(
            label="✨ Character Name",
            placeholder="Chinni"
        )

        clothes = gr.Textbox(
            label="👕 Clothes",
            placeholder="Yellow shirt"
        )

    with gr.Row():

        style = gr.Dropdown(
            [
                "Cute 3D Cartoon",
                "Storybook",
                "Colorful Animation"
            ],
            value="Cute 3D Cartoon",
            label="🎨 Video Style"
        )

        voice_style = gr.Dropdown(
            [
                "Cute Girl",
                "Soft Girl",
                "Playful",
                "Normal"
            ],
            value="Cute Girl",
            label="🎙️ Voice Style"
        )

    create = gr.Button(
        "🎬 CREATE MY VIDEO",
        variant="primary",
        size="lg"
    )

    output = gr.Video(
        label="🎥 Your KAI Video"
    )

    create.click(
        fn=generate_kai_video,
        inputs=[
            story,
            character_type,
            character_name,
            character_color,
            clothes,
            style,
            voice_style
        ],
        outputs=output
    )


app.launch(share=True)
