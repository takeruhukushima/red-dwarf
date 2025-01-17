import json
from fake_useragent import UserAgent
from urllib3.util import ssl_
from requests import Session
from requests.adapters import HTTPAdapter
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin, SQLiteBucket, LimiterSession

ua = UserAgent()

class CachedLimiterSession(CacheMixin, LimiterMixin, Session):
    """
    Session class with caching and rate-limiting behavior. Accepts arguments for both
    LimiterSession and CachedSession.
    """

class CloudflareBypassHTTPAdapter(HTTPAdapter):
    """
    A TransportAdapter that forces TLSv1.3 in Requests, so that Cloudflare doesn't flag us.

    Source: https://lukasa.co.uk/2017/02/Configuring_TLS_With_Requests/
    """

    def init_patched_ssl_context(self):
        context = ssl_.create_urllib3_context()
        context.load_default_certs()
        # Only available in Python 3.7
        if hasattr(ssl_, "TLSVersion"):
            context.minimum_version = ssl_.TLSVersion.TLSv1_3
        else:
            context.options |= ssl_.OP_NO_TLSv1_2

        return context

    def init_poolmanager(self, *args, **kwargs):
        kwargs['ssl_context'] = self.init_patched_ssl_context()
        return super(CloudflareBypassHTTPAdapter, self).init_poolmanager(*args, **kwargs)

    def proxy_manager_for(self, *args, **kwargs):
        kwargs['ssl_context'] = self.init_patched_ssl_context()
        return super(CloudflareBypassHTTPAdapter, self).proxy_manager_for(*args, **kwargs)

class Loader():

    def __init__(self, conversation_id=None, is_cache_enabled=True):
        self.polis_instance_url = "https://pol.is"
        self.conversation_id = conversation_id
        self.is_cache_enabled = is_cache_enabled

        self.votes_data = []
        self.comments_data = []
        self.math_data = {}

        if self.conversation_id:
            self.init_http_client()
            self.load_api_data()

    def init_http_client(self):
        # Throttle requests, but disable when response is already cached.
        if self.is_cache_enabled:
            # Source: https://github.com/JWCook/requests-ratelimiter/tree/main?tab=readme-ov-file#custom-session-example-requests-cache
            self.session = CachedLimiterSession(
                per_second=5,
                cache_name="test_cache.sqlite",
                bucket_class=SQLiteBucket,
                bucket_kwargs={
                    "path": "test_cache.sqlite",
                    'isolation_level': "EXCLUSIVE",
                    'check_same_thread': False,
                },
            )
        else:
            self.session = LimiterSession(per_second=5)
        adapter = CloudflareBypassHTTPAdapter()
        self.session.mount(self.polis_instance_url, adapter)
        self.session.headers = {
            'User-Agent': ua.random,
        }

    def load_api_data(self):
        params = {
            "conversation_id": self.conversation_id,
            "moderation": "true",
            "include_voting_patterns": "true",
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/comments", params=params)
        comments = json.loads(r.text)
        self.comments_data = comments

        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/math/pca2", params=params)
        math = json.loads(r.text)
        self.math_data = math
