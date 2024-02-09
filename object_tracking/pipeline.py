


""" REMOVE detections with confidence too low """

lowest_inds = scores > self.track_low_thresh
bboxes = bboxes[lowest_inds]
scores = scores[lowest_inds]
classes = classes[lowest_inds]
features = output_results[lowest_inds]

""" PICK only high confidence scores """

remain_inds = scores > self.args.track_high_thresh
dets = bboxes[remain_inds]
scores_keep = scores[remain_inds]
classes_keep = classes[remain_inds]
features_keep = features[remain_inds]


""" ??? newly detected tracklets """



""" First association of high score detections using IoU """

# KF filter prediction

# Compute IoU distance

# Matching
matches, u_track, u_detection = matching.linear_assignment(dists, thresh=self.args.match_thresh) # ACTUAL ASSOCIATION
    # u_track, u_detection: unmatched

# KF measurement update for matched ones


""" Second association, with low score detection """

# Pick BBOXs with low scores

# Select unmatched tracklets

# Compute IoU distance

# Matching

# KF measurement update for matched ones

??? '''Deal with unconfirmed tracks, usually tracks with only one beginning frame'''


""" Initialize new tracklets for the unmatched detections that have a score higher than ..."""
if track.score < self.new_track_thresh:
    continue

track.activate(self.kalman_filter, self.frame_id)

""" Remove tracklets that were unmatched for the last X frames """
for track in self.lost_stracks:
    if self.frame_id - track.end_frame > self.max_time_lost:
        track.mark_removed()
        removed_stracks.append(track)
