# Word Sense Disambiguation (WSD) Tool

A powerful, context-aware Word Sense Disambiguation tool built with Streamlit that uses Wikipedia knowledge to understand the meaning of ambiguous words in sentences.

## Features

- **37 Context Types**: Supports programming, biology, finance, entertainment, sports, music, and many more contexts
- **Wikipedia Integration**: Real-time Wikipedia lookups for accurate word definitions
- **Smart Context Detection**: Keyword-based context detection for accurate disambiguation
- **Dark Theme UI**: Modern, clean interface

## Supported Ambiguous Words

| Word | Contexts |
|------|----------|
| python | programming, biology |
| apple | tech_company, food |
| class | programming, education, social |
| file | computer, legal, tools |
| mouse | computer, biology |
| spring | season, water, mechanical |
| crane | construction, bird |
| charge | legal, electrical, payment, military |
| note | writing, music, currency |
| plant | industrial, botany, spy |
| pitch | sports, sales, terrain, music |
| bug | programming, insect, surveillance |
| model | programming, fashion, product |

## Deployment on Streamlit Cloud

1. **Push to GitHub**:
   ```bash
   git init
   git add .
   git commit -m "Initial commit"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/WSD.git
   git push -u origin main
   ```

2. **Deploy on Streamlit Cloud**:
   - Go to [share.streamlit.io](https://share.streamlit.io)
   - Click "New app"
   - Select your repository
   - Set main file path: `app_clean_ui.py`
   - Click "Deploy"

## Local Development

```bash
pip install -r requirements.txt
streamlit run app_clean_ui.py
```

## License

MIT License
