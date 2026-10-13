# Architecture

Each stage is a plain function or small class with data in, data out. Only
`detection/` and `court/keypoints.py` need a GPU model; everything after them is
geometry and logic, covered by `tests/` without any weights
(`Pipeline.analyze` is the model-free half).

## 1. Detection and tracking (`detection/tracker.py`)

One YOLO pass per frame finds every class. Players and referees go through
separate ByteTrack instances (`track_buffer` frames of memory, so a player
screened in the paint keeps their ID). The ball and hoop are not tracked: all
ball candidates are kept for the ball stage, and the most confident hoop box
is kept per frame. An optional second, ball-only model can supply ball
candidates instead.

## 2. Court keypoints, homography, camera (`court/`)

A YOLO pose model predicts 18 landmarks. With 4+ confident, non-collinear
landmarks, `cv2.findHomography` (RANSAC) maps image pixels to court metres.
A fit is rejected if its median reprojection error is above 1 m. Between
good fits the last homography is held for `homography_hold_frames`; consecutive
fits are blended (`homography_smoothing`) unless the court jumps by more than
2 m, which is treated as a camera cut.

`camera_from_homography` recovers the focal length from the two orthogonality
constraints of a plane-induced homography (principal point at the image
centre, square pixels), then rotation and translation. Focal length is median
filtered over time. This turns the 2D floor mapping into a 3x4 camera, which
the 3D stage needs.

Limits: broadcast close-ups often show fewer than 4 landmarks; those frames
have no court mapping. Adding more landmark types (rim, backboard corners,
restricted-area arc points) is the main lever for coverage.

## 3. Ball track (`ball/interpolation.py`)

Per frame, the chosen candidate maximises confidence × closeness to the last
accepted position, so a bright head or a light does not hijack the track.
Detections implying more than `max_speed_px_per_frame` are dropped. Gaps up
to `max_gap_frames` are linearly interpolated; longer gaps stay empty.

## 4. Teams (`teams/team_assigner.py`)

For each track, up to `samples_per_track` torso crops are taken across the
clip. A 2-cluster KMeans inside each crop separates jersey from background
(the cluster owning the crop corners is background). Each track's median
jersey colour (LAB) is then clustered into two teams. One decision per track
means no frame-to-frame flicker. When one team's jersey matches the floor,
switch to `method: clip` with two prompts.

## 5. Possession (`possession/acquisition.py`)

A player is a candidate holder when the ball is close to a "hand point" of
their box, measured as a fraction of box height so it works at any zoom.
Containment halves the distance. A change of holder needs `min_frames`
consecutive frames; losing the ball also needs `min_frames`, which bridges
dribbles.

## 6. Events (`events/`)

Consecutive possession segments with different holders become a pass (same
team), interception (other team, short gap), turnover (other team, long gap),
or rebound (a shot attempt happened in between). Shot attempts are the free
ball entering the zone above the rim; a make is the ball crossing the rim line
downward inside the rim. The shooter is the last holder before the attempt.
Shot location, distance and 2PT/3PT come from the shooter's court position at
release, using the real 3-point arc and corner geometry.

## 7. Court positions, speed, distance (`tactical/`, `metrics/`)

Players are projected by their foot point (bottom-centre of the box).
Positions are median-filtered before differencing; steps faster than
`max_speed_mps` are treated as noise.

## 8. 3D (`recon3d/`)

* Held ball: height is where the camera ray through the ball pixel passes
  closest to the holder's vertical axis.
* Free ball: each run of free-ball frames is fitted as a parabola under
  gravity (initial position and velocity, 6 parameters) by minimising
  reprojection error over all its frames. Runs that do not fit one parabola
  (bounce passes, deflections) are split and refit; implausible fits are
  discarded.
* Players stand on the floor (z = 0) at their projected position.

`scene.json` holds the per-frame scene; `play_3d.html` is an animated Plotly
view. Player height and pose are not modelled yet; a natural next step is a
2D pose model per player box lifted with the recovered camera.

## Roadmap

* Jersey-number OCR to turn track IDs into real players (and re-link IDs after long occlusions).
* More court landmarks for close-up coverage.
* Player pose (skeletons) in the 3D scene.
* Shot outcome from hoop/net motion as a second signal.
* Full-game batching: split by camera cuts, run segments in parallel.
