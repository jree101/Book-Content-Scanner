# Book Content Scanner

**Version 0.1**

A context-aware EPUB scanner identifying objectionable content (profanity, hate speech, violence, sexual content) using AI models and customizable optimization levels.

## Features

- Keyword-based initial filtering with context-aware exclusions
- Transformer models for AI analysis:
  - Toxic-BERT (profanity)
  - RoBERTa Hate Speech
  - Twitter-RoBERTa Sentiment
- **Phase 3** optimization levels (A/B/C) for 8GB VRAM:
  - Option A: Enhanced analysis + violence detector + caching
  - Option B: Option A + 8-bit quantization + batch processing
  - Option C: Option B + 4-bit quantization + quantized Llama Guard
- Custom violence detector
- Smart caching to avoid redundant AI calls
- Configurable entirely via `scanner_settings.py`

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/book-content-scanner.git
   cd book-content-scanner
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Place EPUB files in the `books/` folder.
4. Run the scanner:
   ```bash
   python book_scanner_phase2.py
   ```

## Configuration

Edit `scanner_settings.py`:

- `PHASE3_LEVEL = "A"`  # Change to "B" or "C" for more optimizations
- Adjust `CONFIDENCE_LEVELS` thresholds
- Enable/disable models based on hardware

## Repository Layout

```
book-content-scanner/
|-- .gitignore
|-- LICENSE
|-- README.md
|-- requirements.txt
|-- content_config.py
|-- scanner_settings.py
|-- book_scanner_phase2.py
|-- books/            # Place EPUBs here
|-- scan_reports/     # Generated reports
```

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.