import os
import streamlit as st
from mtranslate import translate
import pandas as pd
from gtts import gTTS
import base64

# Read language dataset using a relative path so it works in any environment
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
df = pd.read_csv(os.path.join(BASE_DIR, 'language.csv'))
df.dropna(inplace=True)
lang = df['name'].to_list()
langlist = tuple(lang)
langcode = df['iso'].to_list()

# Create dictionary of language name → ISO code
lang_array = {lang[i]: langcode[i] for i in range(len(langcode))}

# Layout
st.title("Language Translation")
inputtext = st.text_area("Enter text here to translate", height=100)

choice = st.sidebar.radio('Select Language', langlist)

speech_langs = {
    "af": "Afrikaans",
    "ar": "Arabic",
    "bg": "Bulgarian",
    "bn": "Bengali",
    "bs": "Bosnian",
    "ca": "Catalan",
    "cs": "Czech",
    "cy": "Welsh",
    "da": "Danish",
    "de": "German",
    "el": "Greek",
    "en": "English",
    "eo": "Esperanto",
    "es": "Spanish",
    "et": "Estonian",
    "fi": "Finnish",
    "fr": "French",
    "gu": "Gujarati",
    "od": "Odia",
    "hi": "Hindi",
    "hr": "Croatian",
    "hu": "Hungarian",
    "hy": "Armenian",
    "id": "Indonesian",
    "is": "Icelandic",
    "it": "Italian",
    "ja": "Japanese",
    "jw": "Javanese",
    "km": "Khmer",
    "kn": "Kannada",
    "ko": "Korean",
    "la": "Latin",
    "lv": "Latvian",
    "mk": "Macedonian",
    "ml": "Malayalam",
    "mr": "Marathi",
    "my": "Myanmar (Burmese)",
    "ne": "Nepali",
    "nl": "Dutch",
    "no": "Norwegian",
    "pl": "Polish",
    "pt": "Portuguese",
    "ro": "Romanian",
    "ru": "Russian",
    "si": "Sinhala",
    "sk": "Slovak",
    "sq": "Albanian",
    "sr": "Serbian",
    "su": "Sundanese",
    "sv": "Swedish",
    "sw": "Swahili",
    "ta": "Tamil",
    "te": "Telugu",
    "th": "Thai",
    "tl": "Filipino",
    "tr": "Turkish",
    "uk": "Ukrainian",
    "ur": "Urdu",
    "vi": "Vietnamese",
    "zh-CN": "Chinese"
}


def get_binary_file_downloader_html(bin_file, file_label='File'):
    """Return an HTML anchor tag for downloading a binary file."""
    with open(bin_file, 'rb') as f:
        data = f.read()
    bin_str = base64.b64encode(data).decode()
    href = (
        f'<a href="data:application/octet-stream;base64,{bin_str}" '
        f'download="{os.path.basename(bin_file)}">Download {file_label}</a>'
    )
    return href


c1, c2 = st.columns([4, 3])

if len(inputtext) > 0:
    try:
        output = translate(inputtext, lang_array[choice])
        with c1:
            st.text_area("Translated Text", output, height=200)
        # Only render audio if the target language is supported by gTTS
        if choice in speech_langs.values():
            with c2:
                aud_file = gTTS(text=output, lang=lang_array[choice], slow=False)
                aud_file.save("lang.mp3")
                with open('lang.mp3', 'rb') as audio_file_read:
                    audio_bytes = audio_file_read.read()
                st.audio(audio_bytes, format='audio/mp3')
                st.markdown(
                    get_binary_file_downloader_html("lang.mp3", 'Audio File'),
                    unsafe_allow_html=True
                )
    except Exception as e:
        st.error(str(e))
