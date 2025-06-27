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
            nlp = spacy.load("en_core_web_sm")
            st.success("SpaCy English model 'en_core_web_sm' loaded successfully for sentence tokenization!")
            return nlp
        except OSError:
            st.error("SpaCy English model 'en_core_web_sm' not found.")
            st.info("Please run the following command in your terminal ONCE to download the model, then restart the app:")
            st.code("python -m spacy download en_core_web_sm")
            st.stop()
        except Exception as e:
            st.error(f"An unexpected error occurred while loading spaCy model: {e}")
            st.stop()

# Load the spaCy model at app startup (cached)
spacy_nlp_model = load_spacy_model()

# --- Configuration ---
WHISPER_MODEL_NAME = "base"
WHISPER_COMPUTE_TYPE = "int8"

# --- LLM Configuration for Summarization and Grammar Check ---
LLM_MODEL_NAME = "google/flan-t5-base" # Keep this for now
LLM_MODEL_TYPE = "hf_pipeline" # Keep this for now
SUMMARIZATION_MAX_TOKENS = 150
SUMMARIZATION_MIN_TOKENS = 40
SUMMARIZATION_BEAM_SIZE = 4

# Add a summarization prompt specifically for hf_pipeline
SUMMARIZATION_PROMPT_HF_PIPELINE = "Summarize the following text concisely and without repetition:\n{text}"


# Uncomment for GGUF:
# LLM_MODEL_NAME = "models/gemma-2b-it.Q4_K_M.gguf"
# LLM_MODEL_TYPE = "llama_cpp"
# SUMMARIZATION_MAX_TOKENS = 200
# SUMMARIZATION_MIN_TOKENS = 50
# SUMMARIZATION_PROMPT_TEMPLATE = "<bos><start_of_turn>user\nSummarize the following text:\n{text}<end_of_turn>\n<start_of_turn>model\n"

# --- REVISED GRAMMAR PROMPT ---
GRAMMAR_PROMPT_TEMPLATE = """
As a helpful grammar assistant, please review the following sentence from an audio transcript.
If there are any significant grammatical errors, awkward phrasing, or missing words that make the sentence unnatural, suggest a more natural or grammatically correct way to say it.
Explain the correction briefly and provide the corrected sentence clearly.
If the sentence is perfectly natural and grammatically correct, just say "No changes needed."
Do not repeat the original sentence if no changes are needed.

Examples:
Input Sentence: I was going tomorrow.
Suggestion: You are talking about a future event. Corrected: "I will go tomorrow."

Input Sentence: He don't like apples.
Suggestion: For 'he', 'she', or 'it', use 'doesn't'. Corrected: "He doesn't like apples."

Input Sentence: We went to the store.
Suggestion: No changes needed.

Input Sentence: My.
Suggestion: This is a single word and not a complete sentence that requires grammatical correction. No changes needed.

Input Sentence: So just so that I understand.
Suggestion: This is a common conversational phrase and is grammatically acceptable in context. No changes needed.

Input Sentence: {text}
Suggestion:
"""
GRAMMAR_MAX_TOKENS = 150 # Adjust if needed

# Define a minimum word count for a sentence to be considered for grammar checking
MIN_WORDS_FOR_GRAMMAR_CHECK = 4 # Adjust this threshold as needed (e.g., 2, 3, or 4)

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
                            # Use the new prompt for Hugging Face
                            text_to_summarize = SUMMARIZATION_PROMPT_HF_PIPELINE.format(text=st.session_state.transcript_text)
                            summary_result = llm_model(
                                text_to_summarize, # Use the prepended text
                                max_length=SUMMARIZATION_MAX_TOKENS,
                                min_length=SUMMARIZATION_MIN_TOKENS,
                                do_sample=False,
                                num_beams=SUMMARIZATION_BEAM_SIZE,
                                repetition_penalty=1.2 # Added for better control
                            )
                            summary = summary_result[0]['generated_text']
                        elif LLM_MODEL_TYPE == "llama_cpp":
                            # For llama_cpp, ensure the SUMMARIZATION_PROMPT_TEMPLATE is correctly defined.
                            # It's recommended to define this globally at the top if you intend to use llama_cpp.
                            # If it's only meant to be enabled by uncommenting, then it needs to be accessible here.
                            # For clarity, assuming it would be defined if LLM_MODEL_TYPE is 'llama_cpp'.
                            # If you uncommented the GGUF section, this template would already be in globals().
                            # If not, you might need to define a default here or ensure it's uncommented.
                            if 'SUMMARIZATION_PROMPT_TEMPLATE' not in globals():
                                # Fallback/default if GGUF section wasn't uncommented at the top
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
            # Filter sentences before sending to LLM
            sentences = [
                sent.text.strip() for sent in doc.sents
                if sent.text.strip() and len(sent.text.strip().split()) >= MIN_WORDS_FOR_GRAMMAR_CHECK
            ]

            if not sentences:
                st.info("No sentences found for grammar check (or all were too short).")
            else:
                st.info(f"Checking grammar for {len(sentences)} sentences... This may take a while as each sentence is processed by the LLM.")
                progress_bar = st.progress(0, text="Checking grammar...")

                for i, sentence in enumerate(sentences):
                    # Ensure we don't process empty strings
                    if not sentence:
                        continue

                    grammar_full_prompt = GRAMMAR_PROMPT_TEMPLATE.format(text=sentence) # Use the filtered sentence
                    suggestion_text = "Error generating suggestion."

                    try:
                        if LLM_MODEL_TYPE == "hf_pipeline":
                            grammar_output = llm_model(
                                grammar_full_prompt,
                                # max_length should be large enough for the suggestion + original
                                max_length=len(sentence.split()) + GRAMMAR_MAX_TOKENS,
                                min_length=10, # Ensure a reasonable minimum output length
                                do_sample=False,
                                num_beams=SUMMARIZATION_BEAM_SIZE
                            )
                            raw_suggestion = grammar_output[0]['generated_text'].strip()

                        elif LLM_MODEL_TYPE == "llama_cpp":
                            output = llm_model(
                                grammar_full_prompt,
                                max_tokens=GRAMMAR_MAX_TOKENS,
                                stop=["\nInput Sentence:", "Suggestion:", "</s>", "[/INST]", "<end_of_turn>"],
                                echo=False,
                            )
                            raw_suggestion = output["choices"][0]["text"].strip()
                            # Clean up the output to only get the suggestion part for llama_cpp
                            if raw_suggestion.startswith("Suggestion:"):
                                raw_suggestion = raw_suggestion[len("Suggestion:"):].strip()
                            raw_suggestion = re.split(r'(?i)Input Sentence:', raw_suggestion)[0].strip()

                        # Post-process the raw_suggestion to check for "No changes needed."
                        if "no changes needed" in raw_suggestion.lower():
                            suggestion_text = "No significant issues found." # Use your preferred phrasing
                        else:
                            suggestion_text = raw_suggestion

                    except Exception as e:
                        st.error(f"Error checking grammar for sentence '{sentence[:50]}...': {e}")
                        suggestion_text = f"Error: {e}"

                    st.session_state.grammar_suggestions.append((sentence.strip(), suggestion_text))
                    progress_bar.progress((i + 1) / len(sentences), text=f"Checking grammar for sentence {i+1}/{len(sentences)}...")
                progress_bar.empty()

                if st.session_state.grammar_suggestions:
                    with st.expander("Show Grammar Suggestions"):
                        for original, suggestion in st.session_state.grammar_suggestions:
                            # Only display if there's an actual suggestion or not "No significant issues found."
                            if "No significant issues found." not in suggestion and "Error:" not in suggestion:
                                st.markdown(f"**Original:** `{original}`")
                                st.markdown(f"**Suggestion:** _{suggestion}_")
                                st.markdown("---")
                            elif "Error:" in suggestion:
                                st.markdown(f"**Original:** `{original}`")
                                st.markdown(f"**Error during check:** _{suggestion}_")
                                st.markdown("---")
                            # Optionally, you can display "No significant issues found."
                            # else:
                            #     st.markdown(f"**Original:** `{original}`")
                            #     st.markdown(f"**Suggestion:** _{suggestion}_")
                            #     st.markdown("---")
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

    # Filter grammar suggestions for PDF to only include actual suggestions
    pdf_grammar_suggestions = [
        (original, suggestion) for original, suggestion in st.session_state.grammar_suggestions
        if "No significant issues found." not in suggestion and "Error:" not in suggestion
    ]

    pdf_output_bytes = generate_pdf(
        st.session_state.transcript_text,
        st.session_state.summary_text,
        pdf_grammar_suggestions # Pass filtered suggestions
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