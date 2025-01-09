curl https://pol.is/api/v3/participationInit?conversation_id=4yy3sh84js > sample_data/participationInit.json
curl https://pol.is/api/v3/comments?conversation_id=4yy3sh84js&moderation=true&include_voting_patterns=true > sample_data/comments.json
curl https://pol.is/api/v3/math/pca2?conversation_id=4yy3sh84js > sample_data/math-pca2.json

# TODO: Iterate through all participant IDs and aggregate into one votes file.
# See: https://pol.is/api/v3/votes?conversation_id=4yy3sh84js&pid=0