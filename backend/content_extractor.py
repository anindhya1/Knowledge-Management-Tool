import re

import streamlit as st
from urllib.parse import urlparse, parse_qs
from youtube_transcript_api import YouTubeTranscriptApi
from newspaper import Article


def extract_video_id(url):
    """Extract video ID from various YouTube URL formats"""
    if "youtu.be/" in url:
        video_id = url.split("youtu.be/")[-1].split("?")[0].split("&")[0]
    elif "youtube.com/watch" in url:
        parsed_url = urlparse(url)
        query_params = parse_qs(parsed_url.query)
        video_id = query_params.get('v', [None])[0]
    elif "youtube.com/embed/" in url:
        video_id = url.split("embed/")[-1].split("?")[0]
    else:
        video_id_match = re.search(r'[?&]v=([^&]+)', url)
        video_id = video_id_match.group(1) if video_id_match else None

    if video_id:
        video_id = re.sub(r'[^a-zA-Z0-9_-].*', '', video_id)

    return video_id if video_id and len(video_id) == 11 else None


def get_youtube_transcript(video_id, title):
    """Extract transcript from YouTube video - simplified approach"""
    st.info(f"Extracting transcript for video ID: {video_id}")

    # Simple direct approach - this works for most videos
    transcript_data = YouTubeTranscriptApi.get_transcript(video_id)

    # Combine all transcript text
    transcript_text = " ".join([entry['text'] for entry in transcript_data])

    # Clean up the text
    transcript_text = re.sub(r'\s+', ' ', transcript_text).strip()

    # Format for CSV
    content = f"YouTube Video: {title}\n\nTranscript:\n{transcript_text}"
    content = content.replace('"', "'")  # Escape quotes for CSV

    st.success(f"Successfully extracted transcript ({len(transcript_text)} characters)")
    return content


def extract_article_content(url):
    """Extract text content from a non-YouTube article URL"""
    article = Article(url)
    article.download()
    article.parse()
    return article.text
