# QuoteMiner 🎬

**Transform long-form videos into engaging TikTok clips with AI-powered quote extraction and speaker identification.**

QuoteMiner is an intelligent video processing pipeline that automatically extracts motivational quotes from videos and creates perfectly formatted TikTok clips with speaker diarization, face detection, and optional background video overlays.

## ✨ Features

### 🤖 AI-Powered Quote Extraction

- **Claude AI Integration**: Uses Anthropic's Claude to identify engaging, motivational moments
- **Smart Duration Filtering**: Automatically selects clips between 20-90 seconds
- **Context-Aware Selection**: Identifies standalone, emotionally engaging content perfect for social media

### 🎯 Advanced Video Processing

- **Speaker Diarization**: Identifies who is speaking using pyannote-audio
- **Face Detection**: OpenCV-powered face tracking for optimal cropping
- **TikTok Format**: Automatic conversion to 9:16 aspect ratio (1080x1920)
- **Background Video Support**: Optional background video overlay with smart audio mixing

### ⚡ Performance Optimized

- **Multiprocessing**: Up to 4 parallel workers for fast processing
- **Memory Efficient**: Processes small clips instead of entire videos
- **Mini PC Friendly**: Optimized for systems with 8GB RAM
- **Hardware Adaptive**: Automatic GPU/CPU detection and fallback

### 🛠️ Professional Output

- **High-Quality Videos**: 1080p TikTok-ready clips
- **Smart Audio Mixing**: Preserves voice clarity with 15% background audio
- **Intelligent Naming**: Files named with speaker ID and content preview
- **Automatic Cleanup**: Temporary files managed automatically

## 🚀 Quick Start

### Prerequisites

- Python 3.12+
- [uv](https://docs.astral.sh/uv/) package manager
- [yt-dlp](https://github.com/yt-dlp/yt-dlp) for video downloads (optional)

### Installation

1. **Clone the repository:**

```bash
git clone https://github.com/yourusername/QuoteMiner.git
cd QuoteMiner
```

2. **Install dependencies:**

```bash
uv sync
```

3. **Set up environment variables:**

Create a `.env` file in the project root:

```env
ANTHROPIC_API_KEY=your_anthropic_api_key_here
HUGGINGFACE_TOKEN=your_huggingface_token_here
```

### Basic Usage

#### Process a Single Video

```bash
cd src
uv run python main.py --video-path /path/to/your/video.mp4
```

#### Process Multiple Videos

```bash
uv run python main.py --input-dir /path/to/video/directory/
```

#### Add Background Video

```bash
uv run python main.py --video-path video.mp4 --background-video background.mp4
```

#### Download and Process YouTube Video

```bash
uv run python main.py --video-link "https://youtube.com/watch?v=VIDEO_ID"
```

## 📁 Project Structure

```bash
QuoteMiner/
├── src/
│   ├── config/
│   │   ├── __init__.py
│   │   └── settings.py          # Configuration settings
│   ├── models/
│   │   ├── __init__.py
│   │   ├── quote.py             # Quote data model
│   │   └── video_models.py      # Video processing models
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logger.py            # Logging utilities
│   │   ├── quote_extraction.py  # AI quote extraction
│   │   ├── transcription.py     # Audio transcription
│   │   ├── video_cropper.py     # Video processing pipeline
│   │   └── video_downloader.py  # YouTube video downloads
│   └── main.py                  # Main application entry point
├── data/
│   ├── videos/                  # Input videos
│   ├── quotes/                  # Extracted quotes (JSON)
│   ├── tiktok_clips/           # Output TikTok videos
│   └── transcriptions/         # Audio transcriptions
├── .env                        # Environment variables
├── pyproject.toml              # Project dependencies
└── README.md
```

## 🔧 Advanced Usage

### Direct Video Cropper Usage

For more control over the video processing pipeline:

```bash
cd src
uv run python -m utils.video_cropper \
    /path/to/video.mp4 \
    /path/to/quotes.json \
    /path/to/output/ \
    --background_video /path/to/background.mp4
```

### Configuration Options

Edit `src/config/settings.py` to customize:

- **Hardware Settings**: GPU/CPU preferences, memory constraints
- **Model Selection**: Whisper model size for transcription
- **Processing Limits**: Maximum workers, file paths
- **Quality Settings**: Video resolution, audio quality

### Quote Format

The system expects quotes in JSON format:

```json
[
  {
    "start": "49.44",
    "content": "This is a motivational quote from the video.",
    "end": "58.24"
  }
]
```

## 🎥 Output Examples

### Input

- Long-form podcast or interview video
- YouTube motivational content
- Educational videos with engaging moments

### Output

- Professional TikTok-format clips (9:16 aspect ratio)
- Speaker-focused cropping with face detection
- Clear audio with optional background music
- Filename format: `quote_1_SPEAKER_00_motivational_content.mp4`

## 🔧 Hardware Requirements

### Minimum Requirements

- **CPU**: Multi-core processor (4+ cores recommended)
- **RAM**: 8GB (16GB recommended for batch processing)
- **Storage**: 10GB free space for processing
- **GPU**: Optional (CPU fallback available)

### Optimizations

- **CUDA Support**: Automatic GPU acceleration when available
- **Memory Management**: Efficient processing for resource-constrained systems
- **Parallel Processing**: Scales with available CPU cores

## 🐛 Troubleshooting

### Common Issues

**"No HuggingFace token provided"**

- Ensure `HUGGINGFACE_TOKEN` is set in your `.env` file
- Get a token from [Hugging Face](https://huggingface.co/settings/tokens)

**"ANTHROPIC_API_KEY environment variable not set"**

- Add your Anthropic API key to the `.env` file
- Get an API key from [Anthropic Console](https://console.anthropic.com/)

**Memory Issues**

- Set `CONSTRAINT = True` in `src/config/settings.py`
- Reduce `MAX_WORKERS` to 2 or 1
- Use smaller Whisper models (base.en instead of large-v2)

**Video Processing Errors**

- Ensure input videos are in MP4 format
- Check that OpenCV can access your video files
- Verify sufficient disk space for processing

### Debug Mode

Enable detailed logging by modifying the logger level in `src/main.py`:

```python
logger = Logger(name="QuoteMiner", filename="QuoteMiner.log", level=logging.DEBUG)
```

## 🤝 Contributing

We welcome contributions! Please follow these steps:

1. Fork the repository
2. Create a feature branch: `git checkout -b feature/amazing-feature`
3. Make your changes and test thoroughly
4. Commit with descriptive messages: `git commit -m 'feat: add amazing feature'`
5. Push to your branch: `git push origin feature/amazing-feature`
6. Open a Pull Request

### Development Setup

```bash
# Install development dependencies
uv sync --all-extras

# Run tests (when available)
uv run pytest

# Format code
uv run black src/

# Type checking
uv run mypy src/
```

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **Anthropic** - Claude AI for intelligent quote extraction
- **Hugging Face** - pyannote-audio for speaker diarization
- **OpenAI** - Whisper models for transcription
- **MoviePy** - Video processing capabilities
- **OpenCV** - Computer vision and face detection

## 🔗 Links

- **Documentation**: [Coming Soon]
- **Issues**: [GitHub Issues](https://github.com/yourusername/QuoteMiner/issues)
- **Discussions**: [GitHub Discussions](https://github.com/yourusername/QuoteMiner/discussions)

---

**Made with ❤️ for content creators who want to transform long-form content into engaging short clips.**

*QuoteMiner - Where long videos become viral moments.* ✨
