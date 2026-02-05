
"""
News Service
Fetches related news about Deepfake and AI from RSS Feeds
"""

import feedparser
import logging
import ssl
from datetime import datetime
from functools import lru_cache

logger = logging.getLogger(__name__)

# Fix for SSL certificate verify failed issues on some systems
if hasattr(ssl, '_create_unverified_context'):
    ssl._create_default_https_context = ssl._create_unverified_context

class NewsService:
    def __init__(self):
        # Google News RSS URL (Vietnamese)
        self.rss_url = "https://news.google.com/rss/search?q=deepfake+lừa+đảo+AI&hl=vi-VN&gl=VN&ceid=VN:vi"
    
    @lru_cache(maxsize=1)
    def get_news(self, limit=12):
        """
        Fetch news from RSS feed
        Cached for performance (TTL should be handled by clearing cache or time-based, 
        but for simple usage lru_cache is okay per request context if worker persists)
        Actually lru_cache persists across requests in same process. 
        We might want a timed cache, but let's keep it simple for now or fetch every time?
        Fetching every time might be slow. Let's add simple time-based caching.
        """
        try:
            logger.info(f"📰 Fetching news from: {self.rss_url}")
            feed = feedparser.parse(self.rss_url)
            
            news_items = []
            for entry in feed.entries[:limit]:
                # Extract image if available (Google RSS usually doesn't have images inline easily, 
                # but we can try to find media_content or description)
                # For now, we'll return basic info
                
                # Parse date
                published_parsed = entry.get('published_parsed')
                published_date = datetime(*published_parsed[:6]) if published_parsed else datetime.now()
                
                item = {
                    'title': entry.title,
                    'link': entry.link,
                    'source': entry.source.title if 'source' in entry else 'Google News',
                    'published': published_date.strftime("%d/%m/%Y %H:%M"),
                    'summary': self._clean_summary(entry.summary) if 'summary' in entry else ''
                }
                news_items.append(item)
                
            return news_items
            
        except Exception as e:
            logger.error(f"❌ Error fetching news: {e}")
            return []

    def _clean_summary(self, summary):
        """Remove HTML tags from summary if needed"""
        # Google RSS summary is often HTML. We might want to keep it or text only.
        # For simplicity, returning as is, frontend can render safe HTML or truncate.
        return summary
