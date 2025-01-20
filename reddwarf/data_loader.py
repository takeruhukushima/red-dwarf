import json
import os
from fake_useragent import UserAgent
from urllib3.util import ssl_
from datetime import timedelta
from requests import Session
from requests.adapters import HTTPAdapter
from requests_cache import CacheMixin
from requests_ratelimiter import LimiterMixin, SQLiteBucket, LimiterSession
import csv
from io import StringIO

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

    def __init__(self, conversation_id=None, report_id=None, is_cache_enabled=True, output_dir=None, data_source="api"):
        self.polis_instance_url = "https://pol.is"
        self.conversation_id = conversation_id
        self.report_id = report_id
        self.is_cache_enabled = is_cache_enabled
        self.output_dir = output_dir
        self.data_source = data_source

        self.votes_data = []
        self.comments_data = []
        self.math_data = {}
        self.conversation_data = {}

        if self.conversation_id or self.report_id:
            self.init_http_client()
            if self.data_source == "api":
                self.load_api_data()
            elif self.data_source == "csv_export":
                self.load_remote_export_data()
            else:
                raise ValueError("Unknown data_source: {}".format(self.data_source))

        if self.output_dir:
            self.dump_data(self.output_dir)

    def dump_data(self, output_dir):
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)

        if self.votes_data:
            with open(output_dir + "/votes.json", 'w') as f:
                f.write(json.dumps(self.votes_data, indent=4))

        if self.comments_data:
            with open(output_dir + "/comments.json", 'w') as f:
                f.write(json.dumps(self.comments_data, indent=4))

        if self.math_data:
            with open(output_dir + "/math-pca2.json", 'w') as f:
                f.write(json.dumps(self.math_data, indent=4))

        if self.conversation_data:
            with open(output_dir + "/conversation.json", 'w') as f:
                f.write(json.dumps(self.conversation_data, indent=4))


    def init_http_client(self):
        # Throttle requests, but disable when response is already cached.
        if self.is_cache_enabled:
            # Source: https://github.com/JWCook/requests-ratelimiter/tree/main?tab=readme-ov-file#custom-session-example-requests-cache
            self.session = CachedLimiterSession(
                per_second=5,
                expire_after=timedelta(hours=1),
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

    def load_remote_export_data(self):
        if not self.report_id:
            raise ValueError("Cannot determine CSV export URL without report_id")

        self.load_remote_export_data_comments()
        self.load_remote_export_data_votes()
        # When multiple votes (same tid and pid), keep only most recent (vs first).
        self.filter_duplicate_votes(keep="recent")
        # self.load_remote_export_data_summary()
        # self.load_remote_export_data_participant_votes()
        # self.load_remote_export_data_comment_groups()

    def load_remote_export_data_comments(self):
        r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/comments.csv".format(self.report_id))
        comments_csv = r.text

        reader = csv.DictReader(StringIO(comments_csv))
        COMMENT_FIELD_MAPPING_API_TO_CSV = {
            "created": "timestamp",
            None: "datetime",
            "tid": "comment-id",
            "pid": "author-id",
            "agree_count": "agrees",
            "disagree_count": "disagrees",
            "mod": "moderated",
            "txt": "comment-body",
            "is_seed": "is-seed",
            "is_meta": "is-meta",
            "tweet_id": None,
            "quote_src_url": None,
            "lang": None,
            "velocity": None,
            "active": None,
            "pass_count": None,
            "count": None,
            "conversation_id": None,
        }
        COMMENT_FIELD_MAPPING_CSV_TO_API = {value: key for key, value in COMMENT_FIELD_MAPPING_API_TO_CSV.items()}
        # Make to API fieldnames if a mapping exists, otherwise keep CSV fieldname.
        reader.fieldnames = [(COMMENT_FIELD_MAPPING_CSV_TO_API[f] if COMMENT_FIELD_MAPPING_CSV_TO_API[f] else f) for f in reader.fieldnames]
        self.comments_data = list(reader)

    def load_remote_export_data_votes(self):
        r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/votes.csv".format(self.report_id))
        votes_csv = r.text
        reader = csv.DictReader(StringIO(votes_csv))
        VOTE_FIELD_MAPPING_API_TO_CSV = {
            "modified": "timestamp",
            None: "datetime",
            "tid": "comment-id",
            "pid": "voter-id",
            "vote": "vote",
            "conversation_id": None,
            "weight_x_32767": None,
        }
        VOTE_FIELD_MAPPING_CSV_TO_API = {value: key for key, value in VOTE_FIELD_MAPPING_API_TO_CSV.items()}
        reader.fieldnames = [(VOTE_FIELD_MAPPING_CSV_TO_API[f] if VOTE_FIELD_MAPPING_CSV_TO_API[f] else f) for f in reader.fieldnames]
        self.votes_data = list(reader)

    def filter_duplicate_votes(self, keep="recent"):
        if keep not in {"recent", "first"}:
            raise ValueError("Invalid value for 'keep'. Use 'recent' or 'first'.")

        # Sort by modified time (descending for "recent", ascending for "first")
        if keep == "recent":
            reverse_sort = True
        else:
            reverse_sort = False
        sorted_votes = sorted(self.votes_data, key=lambda x: x["modified"], reverse=reverse_sort)

        filtered_dict = {}
        for v in sorted_votes:
            key = (v["pid"], v["tid"])
            if key not in filtered_dict:
                filtered_dict[key] = v
            else:
                print("Removing duplicate vote: {}".format(v))

        self.votes_data = list(filtered_dict.values())


    def load_remote_export_data_summary(self):
        # r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/summary.csv".format(self.report_id))
        # summary_csv = r.text
        # print(summary_csv)
        raise NotImplementedError

    def load_remote_export_data_participant_votes(self):
        # r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/participant-votes.csv".format(self.report_id))
        # participant_votes_csv = r.text
        # print(participant_votes_csv)
        raise NotImplementedError

    def load_remote_export_data_comment_groups(self):
        # r = self.session.get(self.polis_instance_url + "/api/v3/reportExport/{}/comment-groups.csv".format(self.report_id))
        # comment_groups_csv = r.text
        # print(comment_groups_csv)
        raise NotImplementedError

    def load_api_data(self):
        if self.report_id:
            self.load_api_data_report()
            convo_id_from_report_id = self.report_data["conversation_id"]
            if self.conversation_id and (self.conversation_id != convo_id_from_report_id):
                raise ValueError("report_id conflicts with conversation_id")
            self.conversation_id = convo_id_from_report_id

        self.load_api_data_conversation()
        self.load_api_data_comments()
        self.load_api_data_math()
         # TODO: Add a way to do this without math data, for example
        # by checking until 5 empty responses in a row.
        # This is the best place to check though, as `voters`
        # in summary.csv omits some participants.
        participant_count = self.math_data["n"]
        self.load_api_data_votes(last_participant_id=participant_count-1)

    def load_api_data_report(self):
        params = {
            "report_id": self.report_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/reports", params=params)
        reports = json.loads(r.text)
        self.report_data = reports[0]

    def load_api_data_conversation(self):
        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/conversations", params=params)
        convo = json.loads(r.text)
        self.conversation_data = convo

    def load_api_data_math(self):
        params = {
            "conversation_id": self.conversation_id,
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/math/pca2", params=params)
        math = json.loads(r.text)
        self.math_data = math

    def load_api_data_comments(self):
        params = {
            "conversation_id": self.conversation_id,
            "moderation": "true",
            "include_voting_patterns": "true",
        }
        r = self.session.get(self.polis_instance_url + "/api/v3/comments", params=params)
        comments = json.loads(r.text)
        self.comments_data = comments

    def fix_participant_vote_sign(self):
        """For data coming from the API, vote signs are inverted (e.g., agree is -1)"""
        for item in self.votes_data:
            item["vote"] = -item["vote"]

    def load_api_data_votes(self, last_participant_id=None):
        for pid in range(0, last_participant_id+1):
            params = {
                "pid": pid,
                "conversation_id": self.conversation_id,
            }
            r = self.session.get(self.polis_instance_url + "/api/v3/votes", params=params)
            participant_votes = json.loads(r.text)
            self.votes_data.extend(participant_votes)

        self.fix_participant_vote_sign()
