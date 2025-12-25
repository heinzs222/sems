#!/bin/bash
# =============================================================================
# Build Cached Audio Files
# =============================================================================
# Converts WAV source files to raw mu-law 8kHz format for Twilio.
#
# Usage:
#   bash scripts/build_audio.sh
#
# Prerequisites:
#   - ffmpeg must be installed
#   - WAV files should be in assets/audio/source/
#
# Output:
#   - .mulaw files in assets/audio/
# =============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Directories
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
SOURCE_DIR="$PROJECT_ROOT/assets/audio/source"
OUTPUT_DIR="$PROJECT_ROOT/assets/audio"

echo "==================================="
echo "Building Cached Audio Files"
echo "==================================="
echo ""

# Check for ffmpeg
if ! command -v ffmpeg &> /dev/null; then
    echo -e "${RED}Error: ffmpeg is not installed${NC}"
    echo "Please install ffmpeg first:"
    echo "  - Windows: winget install FFmpeg.FFmpeg"
    echo "  - macOS: brew install ffmpeg"
    echo "  - Ubuntu: sudo apt install ffmpeg"
    exit 1
fi

echo -e "${GREEN}✓ ffmpeg found${NC}"

# Create directories if needed
mkdir -p "$SOURCE_DIR"
mkdir -p "$OUTPUT_DIR"

# List of expected audio files
AUDIO_FILES=("pricing" "who_are_you" "stop")

# Check if source files exist
missing_sources=0
for name in "${AUDIO_FILES[@]}"; do
    source_file="$SOURCE_DIR/${name}.wav"
    if [ ! -f "$source_file" ]; then
        echo -e "${YELLOW}⚠ Missing source file: ${name}.wav${NC}"
        missing_sources=$((missing_sources + 1))
    fi
done

if [ $missing_sources -eq ${#AUDIO_FILES[@]} ]; then
    echo ""
    echo -e "${YELLOW}No source WAV files found in $SOURCE_DIR${NC}"
    echo ""
    echo "To create cached audio responses:"
    echo "1. Record or generate WAV files for each response:"
    echo "   - pricing.wav: Response about your pricing"
    echo "   - who_are_you.wav: Self-introduction response"
    echo "   - stop.wav: Goodbye message"
    echo ""
    echo "2. Place them in: $SOURCE_DIR"
    echo ""
    echo "3. Run this script again"
    echo ""
    echo "Creating placeholder silence files for testing..."
    echo ""
    
    # Create 1-second silence files for testing
    for name in "${AUDIO_FILES[@]}"; do
        output_file="$OUTPUT_DIR/${name}.mulaw"
        if [ ! -f "$output_file" ]; then
            # Create 1 second of silence (8000 bytes of 0xFF for mu-law)
            dd if=/dev/zero bs=8000 count=1 2>/dev/null | tr '\0' '\377' > "$output_file"
            echo -e "${GREEN}Created placeholder: ${name}.mulaw${NC}"
        fi
    done
    
    echo ""
    echo -e "${YELLOW}Note: Placeholder files contain silence. Replace with real audio.${NC}"
    exit 0
fi

# Convert each source file
echo ""
echo "Converting audio files..."
echo ""

converted=0
for name in "${AUDIO_FILES[@]}"; do
    source_file="$SOURCE_DIR/${name}.wav"
    output_file="$OUTPUT_DIR/${name}.mulaw"
    
    if [ -f "$source_file" ]; then
        echo -n "Converting ${name}.wav -> ${name}.mulaw... "
        
        # ffmpeg conversion:
        # -y: Overwrite output
        # -i: Input file
        # -ar 8000: Resample to 8kHz
        # -ac 1: Convert to mono
        # -f mulaw: Output raw mu-law format (no headers)
        ffmpeg -y -i "$source_file" -ar 8000 -ac 1 -f mulaw "$output_file" 2>/dev/null
        
        if [ $? -eq 0 ]; then
            size=$(stat -f%z "$output_file" 2>/dev/null || stat -c%s "$output_file" 2>/dev/null)
            duration_ms=$((size / 8))  # 8 bytes per millisecond at 8kHz mu-law
            echo -e "${GREEN}✓${NC} (${duration_ms}ms)"
            converted=$((converted + 1))
        else
            echo -e "${RED}✗ Failed${NC}"
        fi
    fi
done

echo ""
echo "==================================="
if [ $converted -gt 0 ]; then
    echo -e "${GREEN}Successfully converted $converted file(s)${NC}"
else
    echo -e "${YELLOW}No files converted${NC}"
fi
echo "Output directory: $OUTPUT_DIR"
echo "==================================="
