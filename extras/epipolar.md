# Epipolar geometry: from two views to the disparity–depth relation

## 1. Two cameras, one point

Two pinhole cameras with the same intrinsics

$$
K = \begin{bmatrix} f & 0 & c_x \\ 0 & f & c_y \\ 0 & 0 & 1 \end{bmatrix}
$$

look at a 3D point $X$. With the left camera as the world frame, the right camera is related by a rotation $R$
and translation $t$:

$$
x_L \simeq K\,[\,I \mid 0\,]\,X, \qquad x_R \simeq K\,[\,R \mid t\,]\,X
$$

## 2. The epipolar constraint

Write the normalised image rays $\hat{x}_L = K^{-1}x_L$ and $\hat{x}_R = K^{-1}x_R$. The point $X$, the two
camera centres and both rays lie in one plane (the epipolar plane), so the right ray, the translation and
the rotated left ray are coplanar:

$$
\hat{x}_R \cdot \left(t \times R\,\hat{x}_L\right) = 0
\;\;\Longleftrightarrow\;\;
\hat{x}_R^{\top}\,E\,\hat{x}_L = 0, \qquad E = [t]_\times R
$$

$E$ is the **essential matrix**. In pixels, $x_R^{\top} F\, x_L = 0$ with the **fundamental matrix**
$F = K^{-\top} E K^{-1}$. For a pixel $x_L$, the matching pixel in the right image must lie on the line
$l_R = F x_L$ (its **epipolar line**). Stereo matching therefore searches a line, not the whole image.

## 3. Rectified stereo: epipolar lines become rows

For a rectified rig the cameras are parallel and separated by a baseline $B$ along the $x$-axis:

$$
R = I, \qquad t = (-B, 0, 0)^{\top}
$$

$$
E = [t]_\times = \begin{bmatrix} 0 & 0 & 0 \\ 0 & 0 & B \\ 0 & -B & 0 \end{bmatrix}
\;\;\Rightarrow\;\;
\hat{x}_R^{\top} E\, \hat{x}_L = B\,(\hat{y}_R - \hat{y}_L) = 0
\;\;\Rightarrow\;\; y_R = y_L
$$

A point appears on the **same image row** in both cameras, so the search for a match is one-dimensional.

## 4. Disparity–depth relation

Take $X = (X, Y, Z)$ in the left camera frame. Projecting into both cameras:

$$
x_L = f\,\frac{X}{Z} + c_x, \qquad x_R = f\,\frac{X - B}{Z} + c_x
$$

Subtracting gives the **disparity** $d$:

$$
d = x_L - x_R = \frac{f\,B}{Z}
\qquad\Longrightarrow\qquad
\boxed{\,Z = \frac{f\,B}{d}\,}
$$

This is the same similar-triangles relation as the monocular size method used in this project,
$Z = f\,H/h$, with the known baseline $B$ playing the role of the known object height $H$, and the
disparity $d$ playing the role of the object's pixel height $h$.

## 5. What this means for accuracy

Differentiating with respect to the disparity:

$$
\frac{\partial Z}{\partial d} = -\frac{f B}{d^{2}} = -\frac{Z^{2}}{f B}
\qquad\Rightarrow\qquad
\Delta Z \approx \frac{Z^{2}}{f\,B}\,\Delta d
$$

Depth error grows with the **square** of the distance. Example: $f = 1266$ px (nuScenes front camera),
$B = 0.12$ m (a typical small stereo camera), $\Delta d = 0.5$ px:

| Distance $Z$ | Depth error $\Delta Z$ |
|---|---|
| 2 m | 0.01 m |
| 5 m | 0.08 m |
| 10 m | 0.33 m |
| 20 m | 1.32 m |

The monocular size method behaves the same way: $\Delta Z / Z = \Delta h / h$, so a one-pixel box error on a
cone 20 px tall is already a 5% distance error. A stereo camera removes the dependence on an assumed object
size, which is the main weakness of the monocular approach (a 45 cm cone and a 75 cm cone look the same to it).

## 6. The same geometry in the code

* `distance/estimator.py` — size method $Z = fH/h$ and the ground-plane method, both pinhole projections.
* `distance/nuscenes.py` — `CameraFrame.ground_homography()` builds $H = K\,[\,r_1 \; r_2 \; t\,]$, the
  plane-induced homography between the road plane and the image used for the bird's-eye view.
