"""
SORT: A Simple, Online and Realtime Tracker
Copyright (C) 2016-2020 Alex Bewley alex@bewley.ai

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <http://www.gnu.org/licenses/>.
"""

import numpy as np

from fisheye.enums import TrackingMethod
from fisheye.models.track.base import BaseTracker
from fisheye.models.track.kalman_tracker import KalmanBoxTracker
from fisheye.models.track.utils import associate_detections_to_trackers


class ByteTracker(BaseTracker):
    """ByteTrack Algorithm

    ByteTrack is an IOU-based multi-object tracking (MOT) algorithm that improves association accuracy by incorporating
    both high-confidence and low-confidence detections across frames. The algorithm first associates high-confidence
    detections with existing tracks and then uses unmatched tracks to associate low-confidence detections, reducing
    false negatives and improving long-term tracking stability.
    """

    type = TrackingMethod.BYTETRACK

    def __init__(self, max_age=1, min_hits=3, iou_threshold=0.3):
        """
        Sets key parameters for ByteTrack
        """
        super().__init__(max_age, min_hits, iou_threshold)

    def update(self, dets=(np.empty((0, 5)), np.empty((0, 5)))):
        """
        Params:
        dets - a numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
        Requires: this method must be called once for each frame even with empty detections (use np.empty((0, 5)) for frames without detections).
        Returns a similar array, where the last column is the object ID.

        NOTE: The number of objects returned may differ from the number of detections provided.
        """
        self.frame_count += 1
        # get predicted locations from existing trackers.
        low_dets = dets[0]
        high_dets = dets[1]

        trks = np.zeros((len(self.trackers), 5))
        to_del = []
        ret = []
        for t, trk in enumerate(trks):
            pos = self.trackers[t].predict()[0]
            trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
            if np.any(np.isnan(pos)):
                to_del.append(t)

        trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
        for t in reversed(to_del):
            self.trackers.pop(t)

        high_matched, unmatched_high_dets, unmatched_low_trk_ids = (
            associate_detections_to_trackers(high_dets, trks, self.iou_threshold)
        )
        low_matched, unmatched_low_dets, unmatched_high_trk_ids = (
            associate_detections_to_trackers(low_dets, trks, self.iou_threshold)
        )

        # update matched trackers with assigned detections
        matched_tracks = []
        for m in high_matched:
            self.trackers[m[1]].update(high_dets[m[0], :])
            matched_tracks.append(m[1])
        for m in low_matched:
            if m[1] not in matched_tracks:
                self.trackers[m[1]].update(low_dets[m[0], :])

        # create and initialise new trackers for unmatched detections
        for i in unmatched_high_dets:
            trk = KalmanBoxTracker(high_dets[i, :])
            self.trackers.append(trk)

        i = len(self.trackers)
        for trk in reversed(self.trackers):
            d = trk.get_state()[0]
            if (trk.time_since_update < 1) and (
                trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits
            ):
                ret.append(
                    np.concatenate((d, [trk.id + 1])).reshape(1, -1)
                )  # +1 as MOT benchmark requires positive
            i -= 1
            # remove dead tracklet
            if trk.time_since_update > self.max_age:
                self.trackers.pop(i)

        if len(ret) > 0:
            return np.concatenate(ret)

        return np.empty((0, 5))
