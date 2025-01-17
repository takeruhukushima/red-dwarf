import requests
import json
from requests.adapters import HTTPAdapter
from urllib3.util import ssl_

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

    def __init__(self, conversation_id=None):
        self.polis_instance_url = "https://pol.is"
        self.conversation_id = conversation_id

        self.votes_data = None
        self.comments_data = None
        self.math_data = None

        if self.conversation_id:
            self.init_http_client()
            self.load_api_data()

    def init_http_client(self):
        self.session = requests.Session()
        adapter = CloudflareBypassHTTPAdapter()
        self.session.mount(self.polis_instance_url, adapter)
        self.session.headers = {
            'User-Agent': 'x',
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
