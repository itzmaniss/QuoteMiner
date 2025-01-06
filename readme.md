# QuoteMiner

QuoteMiner is an intelligent video processing tool that harnesses the power of artificial intelligence to automatically discover and extract meaningful quotes from videos. By combining state-of-the-art speech recognition with natural language understanding, it helps content creators and editors transform long-form video content into impactful, shareable moments.

## Understanding How It Works

QuoteMiner operates through a sophisticated pipeline that processes videos in four main stages:

1. First, it uses WhisperX to transcribe the video, creating a precise text representation with word-level timing information. This ensures we know exactly when each word is spoken in the video.

2. Next, it employs the Gemma2 language model through LlamaIndex to analyze the transcript and identify particularly meaningful or impactful quotes. The system looks for passages that contain motivational content, valuable insights, or memorable statements.

3. Then, it matches these extracted quotes back to the original transcript, using advanced text matching algorithms to determine the exact timestamp where each quote appears in the video.

4. Finally, it uses FFmpeg to create precise video clips of these quotes, maintaining the original video and audio quality while extracting just the relevant segments.

## Getting Started

Let's walk through the setup process step by step.

### System Requirements

Before we begin, make sure your system meets these requirements:
- Python 3.9 or higher (needed for modern language features and library compatibility)
- FFmpeg installed and available in system PATH (for video processing)
- CUDA-compatible GPU (optional, but recommended for faster processing)
- Minimum 8GB RAM (16GB recommended for optimal performance)

### Installation

First, clone the repository:
```bash
git clone https://github.com/itzmaniss/QuoteMiner.git
cd QuoteMiner
```

Then, install the required Python packages:
```bash
pip install -r requirements.txt
```

Next, you'll need to install Ollama, which handles our language model:
1. Visit https://ollama.ai/
2. Follow the installation instructions for your operating system
3. Pull the Gemma2 model by running:
```bash
ollama pull gemma2
```

### Configuration

QuoteMiner intelligently adapts to your system's capabilities, but you can fine-tune its behavior through the following settings in `main.py`:

```python
# Set this to True if your system has less than 16GB of RAM
constraint = False  

# The system automatically selects the best model based on your hardware:
# - Systems with GPU: Uses large-v2 model with float16 precision
# - Systems without GPU: Uses small model with int8 precision
# - Systems with RAM constraints: Uses tiny model (when constraint=True)
```

### Using QuoteMiner

Using QuoteMiner is straightforward. The script handles all the necessary setup, including creating the required directory structure. Here's how to use it:

1. Place your MP4 videos in the `videos` directory. The directory will be created automatically when you first run the script.

2. Run the main script:
```bash
python main.py
```

3. QuoteMiner will process each video through these steps:
   - Create transcriptions and store them in the `transcriptions` directory
   - Extract meaningful quotes using AI analysis
   - Generate video clips in the `clips` directory

The output structure will look like this:
```
project_root/
├── videos/           # Put your MP4 files here
├── clips/           # Find your generated clips here
├── transcriptions/  # Contains JSON files with transcripts
└── motivational_lines.json  # List of extracted quotes
```

### Understanding the Output

For each processed video, you'll find:
- A JSON file in the `transcriptions` directory containing the full transcript with timing information
- Multiple video clips in the `clips` directory, named after the original video with numbered suffixes
- A `motivational_lines.json` file listing all extracted quotes

## Advanced Usage and Optimization

QuoteMiner automatically optimizes its performance based on your system's capabilities. Here's how it adapts:

For GPU Systems:
- Utilizes the large-v2 WhisperX model for highest accuracy
- Employs float16 precision to balance speed and memory usage
- Maximizes processing speed through GPU acceleration

For CPU Systems:
- Uses the small WhisperX model to maintain good accuracy
- Implements int8 quantization for efficient processing
- Optimizes memory usage for CPU-based systems

For RAM-Constrained Systems:
- Enable `constraint = True` to use the tiny WhisperX model
- Minimizes memory footprint while maintaining functionality
- Adjusts batch processing size automatically

## Contributing

We welcome contributions to QuoteMiner! If you'd like to help improve the project, please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

QuoteMiner is licensed under the GNU Affero General Public License v3.0 (AGPL-3.0). This means:
- You can use, modify, and distribute the software
- If you modify the software and provide it as a service over a network, you must make your modified source code available
- Any derivative works must also be licensed under AGPL-3.0

For the complete license terms, please see the LICENSE file in the repository.

## Getting Help

If you encounter issues or have questions:
1. Check the existing issues on GitHub
2. Create a new issue if you can't find an answer
3. Provide as much detail as possible about your system and the problem

---

Created with ❤️ for the open-source community by itzmaniss
