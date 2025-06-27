import streamlit as st
from streamlit_mic_recorder import mic_recorder
from faster_whisper import WhisperModel
from transformers import pipeline, AutoTokenizer, AutoModelForSeq2SeqLM
import os
from pydub import AudioSegment
import io
import tempfile
from fpdf import FPDF
import gc
import re
import psutil
import spacy # Import spaCy

# Conditional import for llama_cpp (only if GGUF is intended)
try:
    from llama_cpp import Llama
except ImportError:
    Llama = None # Set to None if not installed, handle gracefully

# --- spaCy Model Loading and Data Check ---
@st.cache_resource
def load_spacy_model():
    """Loads the spaCy English model for sentence tokenization."""
    with st.spinner("Loading spaCy English model for sentence tokenization..."):
        try:
            # Try to load the model
            nlp = spacy.load("en_core_web_sm")
            st.success("SpaCy English model 'en_core_web_sm' loaded successfully for sentence tokenization!")
            return nlp
        except OSError:
            # If model is not found, prompt the user to download it
            st.error("SpaCy English model 'en_core_web_sm' not found.")
            st.info("Please run the following command in your terminal ONCE to download the model, then restart the app:")
            st.code("python -m spacy download en_core_web_sm")
            st.stop() # Stop the app until the model is downloaded
        except Exception as e:
            st.error(f"An unexpected error occurred while loading spaCy model: {e}")
            st.stop()

# Load the spaCy model at app startup (cached)
spacy_nlp_model = load_spacy_model()

# --- Configuration ---
WHISPER_MODEL_NAME = "base"
WHISPER_COMPUTE_TYPE = "int8"

# --- LLM Configuration for Summarization and Grammar Check ---
LLM_MODEL_NAME = "google/flan-t5-base"
LLM_MODEL_TYPE = "hf_pipeline"
SUMMARIZATION_MAX_TOKENS = 150
SUMMARIZATION_MIN_TOKENS = 40
SUMMARIZATION_BEAM_SIZE = 4

# Uncomment for GGUF:
# LLM_MODEL_NAME = "models/gemma-2b-it.Q4_K_M.gguf"
# LLM_MODEL_TYPE = "llama_cpp"
# SUMMARIZATION_MAX_TOKENS = 200
# SUMMARIZATION_MIN_TOKENS = 50
# SUMMARIZATION_PROMPT_TEMPLATE = "<bos><start_of_turn>user\nSummarize the following text:\n{text}<end_of_turn>\n<start_of_turn>model\n"

GRAMMAR_PROMPT_TEMPLATE = """
As a helpful grammar assistant, please review the following sentence from an audio transcript.
If there are any significant grammatical errors or awkward phrasing, suggest a more natural or correct way to say it.
Explain the correction gently and provide a simple example of the correct usage.
If the sentence is perfectly fine, just say "No significant issues found."

Example:
Input Sentence: I was going tomorrow.
Suggestion: It seems like you might be talking about a future event. A more common way to express this would be, "I will go tomorrow."

Input Sentence: He don't like apples.
Suggestion: For 'he', 'she', or 'it', we usually use 'doesn't' instead of 'don't'. So, "He doesn't like apples" would be correct.

Input Sentence: We went to the store.
Suggestion: No significant issues found.

Input Sentence: {text}
Suggestion:
"""
GRAMMAR_MAX_TOKENS = 150

# --- Streamlit Page Setup ---
st.set_page_config(layout="centered", page_title="Audio Processor")
st.title("🗣️ Audio to Transcript & Summary & Grammar Suggestions")
st.markdown("---")

# --- Model Loading Functions ---
@st.cache_resource
def load_whisper_model():
    """Loads the Faster Whisper model, optimized for CPU."""
    with st.spinner(f"Loading Faster Whisper model ({WHISPER_MODEL_NAME}, compute_type='{WHISPER_COMPUTE_TYPE}')..."):
        try:
            model = WhisperModel(WHISPER_MODEL_NAME, device="cpu", compute_type=WHISPER_COMPUTE_TYPE)
            st.success(f"Faster Whisper model '{WHISPER_MODEL_NAME}' loaded successfully with compute_type='{WHISPER_COMPUTE_TYPE}'!")
            return model
        except Exception as e:
            st.error(f"Failed to load Faster Whisper model. Please check your internet connection and disk space. Error: {e}")
            st.stop()

@st.cache_resource
def load_llm_for_tasks(model_name, model_type):
    """
    Loads a language model for summarization and potentially grammar correction.
    Handles both Hugging Face pipelines and Llama.cpp GGUF models.
    """
    with st.spinner(f"Loading LLM ({model_name})..."):
        try:
            if model_type == "hf_pipeline":
                tokenizer = AutoTokenizer.from_pretrained(model_name)
                model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
                llm_pipeline = pipeline("text2text-generation", model=model, tokenizer=tokenizer, device=-1)
                st.success(f"Hugging Face LLM '{model_name}' loaded successfully!")
                return llm_pipeline
            elif model_type == "llama_cpp":
                if Llama is None:
                    st.error("`llama-cpp-python` is not installed. Please install it to use GGUF models.")
                    st.stop()
                if not os.path.exists(model_name):
                    st.error(f"GGUF model not found at {model_name}. Please download it and place it in the 'models' folder.")
                    st.stop()
                llm_model = Llama(model_path=model_name, n_ctx=2048, n_batch=512, n_threads=os.cpu_count(), verbose=False)
                st.success(f"Llama.cpp GGUF model '{model_name}' loaded successfully!")
                return llm_model
            else:
                st.error(f"Unknown LLM model type: {model_type}. Please check configuration.")
                st.stop()
        except Exception as e:
            st.error(f"Failed to load LLM. Please check your internet connection, system resources, and ensure all libraries are up-to-date. Error: {e}")
            st.stop()

# Load models at app startup (cached)
whisper_model = load_whisper_model()
llm_model = load_llm_for_tasks(LLM_MODEL_NAME, LLM_MODEL_TYPE)

# --- Session state for audio and results ---
if "audio_data_for_processing" not in st.session_state:
    st.session_state.audio_data_for_processing = None
if "transcript_text" not in st.session_state:
    st.session_state.transcript_text = None
if "summary_text" not in st.session_state:
    st.session_state.summary_text = None
if "grammar_suggestions" not in st.session_state:
    st.session_state.grammar_suggestions = []

st.markdown("---")
st.header("1. Choose Audio Input Method")

audio_source_option = st.radio(
    "Select input method:",
    ("Upload Audio File", "Record Live Audio"),
    key="audio_source_radio"
)

# --- Upload Audio File ---
if audio_source_option == "Upload Audio File":
    uploaded_file = st.file_uploader(
        "Upload an audio file (.mp3, .wav, .m4a, .flac)",
        type=["mp3", "wav", "m4a", "flac"]
    )

    if uploaded_file:
        st.audio(uploaded_file, format=uploaded_file.type)
        temp_path = None

        try:
            original_suffix = os.path.splitext(uploaded_file.name)[1]
            with tempfile.NamedTemporaryFile(delete=False, suffix=original_suffix) as temp_upload_file:
                temp_upload_file.write(uploaded_file.read())
                temp_path = temp_upload_file.name

            audio_segment = AudioSegment.from_file(temp_path)
            temp_wav_io = io.BytesIO()
            audio_segment.export(temp_wav_io, format="wav")
            temp_wav_io.seek(0)
            st.session_state.audio_data_for_processing = temp_wav_io
            st.info(f"Uploaded audio length: {audio_segment.duration_seconds:.2f} sec")
        except Exception as e:
            st.error(f"Error processing uploaded file: {e}. Please ensure FFmpeg is installed and the file is not corrupted.")
            st.session_state.audio_data_for_processing = None
        finally:
            if temp_path and os.path.exists(temp_path):
                os.remove(temp_path)

# --- Record Live Audio ---
elif audio_source_option == "Record Live Audio":
    st.info("Click the mic to start/stop recording. Auto stops after 30s.")
    recorded = mic_recorder(
        start_prompt="Start Recording",
        stop_prompt="Stop Recording",
        just_once=True,
        use_container_width=True,
        key="mic_recorder"
    )

    if recorded and 'bytes' in recorded:
        audio_bytes = recorded['bytes']
        st.audio(audio_bytes, format="audio/wav")

        audio_buffer = io.BytesIO(audio_bytes)
        audio_buffer.seek(0)
        st.session_state.audio_data_for_processing = audio_buffer
        st.success("Audio recorded successfully!")
    elif recorded is None:
        st.info("No audio recorded yet.")
    else:
        st.warning("Recording failed or was empty.")
        st.session_state.audio_data_for_processing = None

st.markdown("---")

# --- Function to generate PDF ---
class PDF(FPDF):
    def header(self):
        self.set_font("Arial", "B", 12)
        self.cell(0, 10, "Audio Processing Results", 0, 1, "C")
        self.ln(5)

    def footer(self):
        self.set_y(-15)
        self.set_font("Arial", "I", 8)
        self.cell(0, 10, f"Page {self.page_no()}/{{nb}}", 0, 0, "C")

    def add_section(self, title, content):
        self.set_font("Arial", "B", 12)
        self.multi_cell(0, 10, f"--- {title} ---")
        self.ln(2)
        self.set_font("Arial", "", 10)
        try:
            self.multi_cell(0, 7, content)
        except UnicodeEncodeError:
            st.warning(f"Some characters in the {title.lower()} could not be rendered in the PDF (due to font limitations). They have been replaced with '?').")
            self.multi_cell(0, 7, content.encode('latin-1', 'replace').decode('latin-1'))
        self.ln(5)

def generate_pdf(transcript, summary, grammar_suggestions):
    pdf = PDF()
    pdf.add_page()
    pdf.alias_nb_pages()

    pdf.add_section("Transcript", transcript if transcript else "No transcript available.")

    if summary:
        pdf.add_section("Summary", summary)

    if grammar_suggestions:
        grammar_text = ""
        for i, (original, suggestion) in enumerate(grammar_suggestions):
            grammar_text += f"Original: {original}\nSuggestion: {suggestion}\n\n"
        pdf.add_section("Grammar Suggestions", grammar_text if grammar_text else "No grammar suggestions.")

    return pdf.output(dest='S').encode('latin-1')

# --- Process Audio ---
if st.button("Process Audio", use_container_width=True):
    st.session_state.transcript_text = None
    st.session_state.summary_text = None
    st.session_state.grammar_suggestions = []

    if st.session_state.audio_data_for_processing:
        st.subheader("Processing Results:")

        # --- Transcribe Audio ---
        with st.spinner("Transcribing audio... (This may take a while on CPU)"):
            try:
                st.session_state.audio_data_for_processing.seek(0)
                segments, info = whisper_model.transcribe(st.session_state.audio_data_for_processing, beam_size=5)
                transcript = " ".join([s.text.strip() for s in segments if s.text.strip()])

                st.session_state.transcript_text = transcript
                st.success(f"Transcription Complete! Detected language: {info.language.upper()}")
                st.subheader("📝 Transcript:")
                st.code(transcript)

            except Exception as e:
                st.error(f"Transcription failed: {e}. Please check the audio file and ensure Whisper model loaded correctly.")
                st.session_state.transcript_text = None

        # --- Summarize Transcript ---
        if st.session_state.transcript_text:
            if len(st.session_state.transcript_text.split()) >= 30:
                with st.spinner("Generating summary... (This may take a while on CPU)"):
                    try:
                        summary = ""
                        if LLM_MODEL_TYPE == "hf_pipeline":
                            summary_result = llm_model(
                                st.session_state.transcript_text,
                                max_length=SUMMARIZATION_MAX_TOKENS,
                                min_length=SUMMARIZATION_MIN_TOKENS,
                                do_sample=False,
                                num_beams=SUMMARIZATION_BEAM_SIZE
                            )
                            summary = summary_result[0]['generated_text']
                        elif LLM_MODEL_TYPE == "llama_cpp":
                            if 'SUMMARIZATION_PROMPT_TEMPLATE' not in globals():
                                SUMMARIZATION_PROMPT_TEMPLATE = "<bos><start_of_turn>user\nSummarize the following text:\n{text}<end_of_turn>\n<start_of_turn>model\n"

                            prompt = SUMMARIZATION_PROMPT_TEMPLATE.format(text=st.session_state.transcript_text)
                            output = llm_model(
                                prompt,
                                max_tokens=SUMMARIZATION_MAX_TOKENS,
                                stop=["</s>", "[/INST]", "<end_of_turn>"],
                                echo=False,
                            )
                            summary = output["choices"][0]["text"].strip()

                        st.session_state.summary_text = summary
                        st.success("Summary Complete!")
                        st.subheader("📄 Summary:")
                        st.write(summary)
                    except Exception as e:
                        st.error(f"Summarization failed: {e}. Try adjusting length parameters or using a shorter transcript.")
                        st.session_state.summary_text = None
            else:
                st.warning("Transcript too short for meaningful summary (min 30 words). Skipping summarization.")
                st.session_state.summary_text = None
        else:
            st.warning("Skipping summarization as no transcript was generated.")

        # --- Grammatical Error Suggestions ---
        if st.session_state.transcript_text and LLM_MODEL_TYPE in ["hf_pipeline", "llama_cpp"]:
            st.subheader("✍️ Grammar Suggestions:")

            # Use spaCy for sentence tokenization
            doc = spacy_nlp_model(st.session_state.transcript_text)
            sentences = [sent.text.strip() for sent in doc.sents if sent.text.strip()] # Extract and clean sentences

            if not sentences:
                st.info("No sentences found for grammar check.")
            else:
                st.info(f"Checking grammar for {len(sentences)} sentences... This may take a while as each sentence is processed by the LLM.")
                progress_bar = st.progress(0, text="Checking grammar...")

                for i, sentence in enumerate(sentences):
                    if not sentence.strip():
                        continue

                    grammar_full_prompt = GRAMMAR_PROMPT_TEMPLATE.format(text=sentence.strip())
                    suggestion_text = "Error generating suggestion."

                    try:
                        if LLM_MODEL_TYPE == "hf_pipeline":
                            grammar_output = llm_model(
                                grammar_full_prompt,
                                max_length=len(sentence.split()) + GRAMMAR_MAX_TOKENS,
                                min_length=10,
                                do_sample=False,
                                num_beams=SUMMARIZATION_BEAM_SIZE
                            )
                            suggestion_text = grammar_output[0]['generated_text'].strip()

                        elif LLM_MODEL_TYPE == "llama_cpp":
                            output = llm_model(
                                grammar_full_prompt,
                                max_tokens=GRAMMAR_MAX_TOKENS,
                                stop=["\nInput Sentence:", "Suggestion:", "</s>", "[/INST]", "<end_of_turn>"],
                                echo=False,
                            )
                            suggestion_text = output["choices"][0]["text"].strip()
                            if suggestion_text.startswith("Suggestion:"):
                                suggestion_text = suggestion_text[len("Suggestion:"):].strip()
                            suggestion_text = re.split(r'(?i)Input Sentence:', suggestion_text)[0].strip()

                    except Exception as e:
                        st.error(f"Error checking grammar for sentence '{sentence[:50]}...': {e}")
                        suggestion_text = f"Error: {e}"

                    st.session_state.grammar_suggestions.append((sentence.strip(), suggestion_text))
                    progress_bar.progress((i + 1) / len(sentences), text=f"Checking grammar for sentence {i+1}/{len(sentences)}...")
                progress_bar.empty()

                if st.session_state.grammar_suggestions:
                    with st.expander("Show Grammar Suggestions"):
                        for original, suggestion in st.session_state.grammar_suggestions:
                            st.markdown(f"**Original:** `{original}`")
                            st.markdown(f"**Suggestion:** _{suggestion}_")
                            st.markdown("---")
                else:
                    st.info("No grammar suggestions generated.")
        else:
            st.warning("Skipping grammar check as no transcript was generated or no LLM configured for it.")

    else:
        st.warning("Please upload or record audio before processing.")

# --- Download Buttons ---
if st.session_state.transcript_text or st.session_state.summary_text or st.session_state.grammar_suggestions:
    st.markdown("---")
    st.header("Download Results")

    pdf_output_bytes = generate_pdf(
        st.session_state.transcript_text,
        st.session_state.summary_text,
        st.session_state.grammar_suggestions
    )

    st.download_button(
        label="Download Transcript, Summary & Grammar as PDF",
        data=pdf_output_bytes,
        file_name="audio_analysis.pdf",
        mime="application/pdf",
        use_container_width=True
    )

    if st.session_state.transcript_text:
        st.download_button(
            label="Download Transcript as TXT",
            data=st.session_state.transcript_text.encode('utf-8'),
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True
        )

st.markdown("---")
st.caption(f"Whisper Model: {WHISPER_MODEL_NAME.capitalize()} (Compute: {WHISPER_COMPUTE_TYPE}), LLM for Summarization/Grammar: {LLM_MODEL_NAME}")