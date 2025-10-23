import os, uuid, tempfile, asyncio
import gradio as gr
from openai import OpenAI
from agents import Agent, Runner, function_tool, SQLiteSession

# ---- APIキー（HF Spacesの「Settings > Variables and secrets」で設定）----
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("Environment variable OPENAI_API_KEY is not set.")
client = OpenAI(api_key=api_key)

# ---- STT/TTS --------------------------------------------------------------
async def speech_to_text(audio_path: str) -> str:
    with open(audio_path, "rb") as f:
        tr = client.audio.transcriptions.create(
            model="gpt-4o-mini-transcribe",
            file=f,
        )
    return (tr.text or "").strip()

async def text_to_speech(text: str, voice: str = "alloy") -> str:
    out_path = os.path.join(tempfile.gettempdir(), f"reply_{uuid.uuid4().hex}.mp3")
    with client.audio.speech.with_streaming_response.create(
        model="gpt-4o-mini-tts",
        voice=voice,
        input=text,
    ) as resp:
        resp.stream_to_file(out_path)
    return out_path

# ---- 関数ツール（デモ用）---------------------------------------------------
TODO = []

@function_tool
def add_todo(task: str) -> str:
    if not task.strip():
        return "空のタスクは追加できません。"
    TODO.append(task.strip())
    return f"タスクを追加: {task}（合計 {len(TODO)} 件）"

@function_tool
def list_todo() -> list[str]:
    return TODO

@function_tool
def clear_todo() -> str:
    TODO.clear()
    return "タスクをすべて削除しました。"

@function_tool
def now() -> str:
    from datetime import datetime, timezone, timedelta
    JST = timezone(timedelta(hours=9), name="JST")
    return datetime.now(JST).strftime("%Y-%m-%d %H:%M")

# ---- エージェント & セッション -------------------------------------------
secretary = Agent(
    name="Voice Secretary",
    instructions=(
        "あなたは音声でやりとりする日本語の秘書です。"
        "丁寧でわかりやすく、1〜3文で簡潔に答えてください。"
        "最後に『次のアクション』を1つ提案します。"
        "必要に応じて add_todo / list_todo / clear_todo / now を使ってください。"
    ),
    tools=[add_todo, list_todo, clear_todo, now],
)
session = SQLiteSession("voice_secretary_space")

GREETING = "こんにちは。秘書のエコです。ご用件をどうぞ。"

async def handle_interaction(audio_file, text_input, voice, messages):
    messages = messages or []
    # 入力を取得（音声優先）
    if audio_file:
        try:
            user_text = await speech_to_text(audio_file)
        except Exception as e:
            user_text = ""
            messages.append({"role":"assistant","content":f"文字起こしエラー: {e}"})
    else:
        user_text = (text_input or "").strip()

    if not user_text:
        messages.append({"role":"assistant","content":"音声またはテキストで話しかけてください。"})
        return messages, None

    messages.append({"role":"user","content":user_text})

    # エージェント実行
    try:
        result = await Runner.run(secretary, input=user_text, session=session)
        bot_text = (result.final_output or "").strip()
    except Exception as e:
        bot_text = f"回答生成でエラーが発生しました: {e}"

    messages.append({"role":"assistant","content":bot_text})

    # TTS
    tts_path = None
    try:
        tts_path = await text_to_speech(bot_text, voice=voice)
    except Exception as e:
        messages.append({"role":"assistant","content":f"音声合成に失敗しました: {e}"})

    return messages, tts_path

async def clear_all():
    TODO.clear()
    try:
        await session.clear_session()
    except Exception:
        pass
    return [{"role":"assistant","content":GREETING}], None

# ---- Gradio UI ------------------------------------------------------------
with gr.Blocks(title="音声秘書アプリ") as demo:
    gr.Markdown(
        "## 🎧 音声でやりとりする秘書アプリ\n"
        "- マイクで話しかけると、秘書がテキストと音声で返答します。\n"
        "- 例：「午後3時に資料送付をリマインド」「タスクを一覧して」など。"
    )

    with gr.Row():
        audio_in = gr.Audio(sources=["microphone"], type="filepath", label="🎤 マイク入力")
        text_in = gr.Textbox(label="⌨️ テキスト入力（音声が使えないとき）")

    with gr.Row():
        voice = gr.Dropdown(choices=["alloy","shimmer","nova","onyx","echo","fable"],
                            value="alloy", label="音声の種類（TTS Voice）")
        send_btn = gr.Button("▶️ 送信", variant="primary")
        clear_btn = gr.Button("🧹 クリア")

    chat = gr.Chatbot(
        label="会話",
        type="messages",
        value=[{"role":"assistant","content":GREETING}],
        height=360,
    )
    audio_out = gr.Audio(label="🔊 音声回答", autoplay=True)
    state = gr.State([{"role":"assistant","content":GREETING}])

    send_btn.click(
        fn=handle_interaction,
        inputs=[audio_in, text_in, voice, state],
        outputs=[chat, audio_out]
    ).then(fn=lambda h, *_: h, inputs=[chat], outputs=[state])

    clear_btn.click(
        fn=clear_all, inputs=[], outputs=[chat, audio_out]
    ).then(fn=lambda h, *_: h, inputs=[chat], outputs=[state])

# Spaces ではポート/ホストは自動設定されます
if __name__ == "__main__":
    demo.launch()
