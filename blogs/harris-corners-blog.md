---
title: "Harris Corner Detection"
date: "December 2024"
description: "An in-depth exploration of the Harris corner detection algorithm, covering the mathematical foundations, implementation details, and practical applications in computer vision."
tags: ["Computer Vision", "Corner Detection", "Image Processing", "Algorithms"]
---

# Harris Corners

## Problem Definition

Our goal is to design a function whose shape will tell us whether a point in an image is a corner or not.

To begin designing this function let's first consider a particular point $\underline{P}$ and only look at a small window $W$ around that point. We want to design a function that will tell us whether our point is a good Keypoint (corner) or not.

*[Image placeholder: Window around point P]*

Suppose we take another window $W'$ that's centered at a different point $\underline{P}'$ that is very close to $\underline{P}$. That is if $\underline{P} = [x, y]^T$ then $\underline{P}' = [x+u, y+v]^T$ for some small $(u,v)$. Now if we compare the two windows by summing the square differences between corresponding pixels and find that the summation is **large** then this might lead us to thinking that $\underline{P}$ is a good keypoint.

Why? Because moving the window by just a little bit made a big difference which is essentially what a definition keypoint could be. That is a Keypoint could be defined as a region around point that is unique in an image and therefore comparing this region with any other region (near or far) will give us a relatively large difference.

If we repeat this process for different small shifts $(u,v)$ and find that we are always getting a large difference or error then we have found a good Keypoint.

*[Image placeholder: Different shifts around keypoint]*

Going back to the function we want to design, we want a function $E(u,v)$ that will tell us the difference between a window centered at a particular point $\underline{P}$ and a window centered at a point shifted by $(u,v)$ from $\underline{P}$. Mathematically, this desired function is:

$$E(u,v) = \sum_{x,y} W(x,y) [I(x+u, y+v) - I(x, y)]^2$$

where

$$w(x,y) = \begin{cases} 1, & \text{if } (x,y) \in W \\ 0, & \text{Otherwise} \end{cases}$$

Our claim so far is as follows: If we find that the difference between a window centered at a particular point $\underline{P}$ and every other window centered at any point $\underline{P}'$ that is shifted by any small $(u,v)$ then we will consider P to be a Keypoint.

Let's look at some examples that back this claim.

*[Image placeholder: Examples of corners vs edges vs flat regions]*

## Approximation

Now we know what we are looking for (points in our image where $E(u,v)$ is large for any small $(u,v)$ in the window around that point).

This is actually decently computationally intensive as for every candidate point $\underline{P}$ in our image we are computing $E(u,v)$ for over some range $(u,v) \in W$. The run-time for this process would be $O(KN)$ where $K = \text{(window width)} \times \text{(window height)}$ and $N$ is the number of pixels in the image. Additionally, at this point we will have only computed an $E(\cdot, \cdot)$ matrix for every candidate point $\underline{P}$ but will not have decided which points are good corners and which are not. This is the motivation for what follows.

Instead of actually computing $E(u,v) \quad \forall (u,v) \in W$ at every point or pixel location $\underline{P}$, we will approximate the function $E(u,v)$ by a second order taylor polynomial centered at that particular point.

If we do this we obtain:

$$E(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}$$

So we can compute $E(u,v)$ for any $(u,v)$ shift from a particular point $\underline{P}$. Furthermore, our error function looks like this:

*[Image placeholder: 3D visualization of quadratic error function]*

Now, previously we claimed that if $E(u,v)$ is large for all small $(u,v)$ then P is a corner. We are getting closer to an algorithm. Let's consider all $(u,v)$ shifts that give rise to some particular error, that is let's look at a slice of the error function of constant error $K$:

$$E(u,v) = k = \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}$$

we notice that is a quadratic form and in particular it is an equation of an ellipse. It can be shown that given an ellipse equation written in quadratic form, the major and minor axis lengths of an ellipse are given by:

*[Image placeholder: Ellipse with axis lengths marked]*

where $\lambda_1$ and $\lambda_2$ are the eigenvalues of matrix $M$. We will see why this is the case in just a bit. For now take it as a property of any ellipse equation.

We now notice that if we consider all shifts $(u,v)$ that cause the same error $K = E(u,v)$. A **larger** $(u,v)$ shift is needed if the eigenvalues of the matrix M are **small**, as the radius of the ellipse is **large**.

*[Image placeholder: Comparison of ellipses with different eigenvalues]*

More concretely, suppose we are comparing two candidate points $\underline{P_1}$ and $\underline{P_2}$ and find their corresponding error functions to be:

$$E_1(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M_1 \begin{bmatrix} u \\ v \end{bmatrix}$$

$$E_2(u,v) \approx \begin{bmatrix} u & v \end{bmatrix} M_2 \begin{bmatrix} u \\ v \end{bmatrix}$$

But we find that the eigenvalues of $M_1$ are larger than the eigenvalues of $M_2$ then we can say that $\underline{P_1}$ is a better keypoint because a **smaller** $(u,v)$ shift causes the **same** error as a **larger** $(u,v)$ shift does at $\underline{P_2}$.

## Principal Axes Theorem

Let $Q$ be a quadratic form on $\mathbb{R}^{N}$ (i.e. $Q(\underline{x}) = \underline{x}^T A \underline{x}$) where $A \in \mathbb{R}^{N \times N}$ is a symmetric matrix. Then we can find an **orthogonal** matrix $P \in \mathbb{R}^{N \times N}$ such that $\underline{x}= P\underline{y}$, transforming $Q(\underline{x}) = \underline{x}^T A \underline{x}$ to $Q(\underline{x}) = \underline{y}^T D \underline{y}$ with no cross terms. ($D$ is a diagonal matrix)

So why are the radii of the ellipse given by the $r_i = \lambda_i^{1/2}$? This is kind of a long story. First we start with our original equation. Let $\underline{x} = \begin{bmatrix} x_1 & x_2 \end{bmatrix}^T$ and suppose we have the following ellipse equation.

$$K = \underline{x}^T M \underline{x} = a_1 u^2 + a_2 uv + a_3 v^2$$

we will need three facts to proceed:

1. **Matrix $\mathbf{M}$ is symmetric** and there is a theorem that states that we can always diagonalize a symmetric matrices - that is we can find a matrix $P$ such that $P^T M P = D$ where $D$ is a diagonal matrix and the columns of $P$ are the eigenvectors of $M$.

2. **Change of basis**: Another thing we'll need is to recall that any invertible matrix $\mathbf{P} \in \mathbb{R}^{N \times N}$ represents a change of basis from $\eta$-coordinates to $\beta$-coordinates:
   
   $$\underline{x} = P \underline{y} \quad \Longleftrightarrow \quad \underline{y} = P^{-1} \underline{x} \quad \Longleftrightarrow \quad [\underline{y}]_\beta = [\underline{x}]_\eta$$

3. **Standard Ellipse equation**: The equation of a Standard Ellipse gives an ellipse that is symmetric about the axes.
   
   $$\frac{x^2}{r_1^2} + \frac{y^2}{r_2^2} = k$$
   
   Or in vector form this is:
   
   $$\begin{bmatrix} x & y \end{bmatrix} \begin{bmatrix} 1/r_1^2 & 0 \\ 0 & 1/r_2^2 \end{bmatrix} \begin{bmatrix} x \\ y \end{bmatrix} = k$$
   
   *[Image placeholder: Standard ellipse aligned with axes]*
   
   Where the radii of this ellipse is then given by $r_1$ and $r_2$. We should note that the more General Equation of an Ellipse does not give us an ellipse symmetric about the axes due to the cross term $bxy$.
   
   $$ax^2 + bxy + cy^2 = k$$
   
   *[Image placeholder: Rotated ellipse with cross terms]*

Now we can proceed as follows. Suppose we define a change of basis matrix $P$, and now $x$ can be given by:

$$\underline{x} = P \underline{y}$$

where $\underline{x}$ is in standard coordinates and $\underline{y}$ is in a different coordinate system where the columns of $P$ are the basis vectors of that coordinate system. We can then re-write our ellipse equation:

$$K = \underline{x}^T M \underline{x} = \underline{y}^T P^T A P \underline{y} = \underline{y}^T C \underline{y}$$

Now we observe two things. First that $K = \underline{y}^T C \underline{y}$ defines the exact same ellipse in space as $K = \underline{x}^T M \underline{x}$. The only difference is the set of all points that make up the ellipse are referenced from a different coordinate system (different basis). Second, $P$ can be any arbitrary change of basis matrix and nothing interesting would happen, we'd be left with this new equation for the same ellipse:

$$K = \underline{y}^T C \underline{y} = ay_1^2 + by_1y_2 + cy_2^2$$

However, if instead we choose a change of basis matrix $P$ whose columns are the eigenvectors of A then by **Fact #2** we see that $C = P^T A P = D$ is a **diagonal** matrix. Expanding this out we get:

$$K = \underline{y}^T D \underline{y} = \begin{bmatrix} y_1 & y_2 \end{bmatrix} \begin{bmatrix} \lambda_1 & 0 \\ 0 & \lambda_2 \end{bmatrix} \begin{bmatrix} y_1 \\ y_2 \end{bmatrix} = \lambda_1y_1^2 + \lambda_2 y_2^2$$

So by changing our reference frame we've just eliminated our cross term! We can then use **Fact #3** and match coefficients. If we do this we find the following:

$$\frac{y_1^2}{r_1^2} + \frac{y_2^2}{r_2^2} = k = \lambda_1y_1^2 + \lambda_2 y_2^2$$

$$\therefore r_1 = \lambda_1^{-1/2} \text{ and } r_2 = \lambda_2^{-1/2}$$

## Algorithm

By now we know that a particular pixel $\underline{P}$ is categorized as a Keypoint if when we compute an error function $E(u,v)$ and take any particular contour $E(u,v) = k$ we see a narrow ellipse. This corresponds to the fact that if we took a small window around that point and shifted it by a little bit then compared it to the original window we'd see a relatively large difference.

We have also seen that we can quantify the "narrow-ness" of a particular contour or ellipse of $E(u,v)$ by using the eigenvalues of the matrix $M$ in our quadratic form equation:

$$E(u,v) = \begin{bmatrix} u & v \end{bmatrix} M \begin{bmatrix} u \\ v \end{bmatrix}$$

So we want to find points in the image that give rise to a quadratic where the $M$ matrix has large eigenvalues. We also want the eigenvalues to be comparable in size. Why? Well if one of the eigenvalues is small this means that the ellipse we are looking at is narrow in one direction and long in the other and therefore our error $E(u,v)$ is high if we shift the window around our point in one direction but is low if we shift it in another direction. This means we are at a line.

In the original paper "A COMBINED CORNER AND EDGE DETECTOR" the author's proposed the following "corner-ness" score to find points with large eigenvalues that are close in magnitude:

$$R = \lambda_1 \lambda_2 - \alpha(\lambda_1 + \lambda_2)^2$$

To interpret this formulation, let's consider two cases. If $\lambda_2$ is large and $\lambda_1$ is small and $\lambda_2 = 10\lambda_1$ and we take $\alpha = 0.05$:

$$R = \lambda_1 (10\lambda_1) - \alpha(\lambda_1 + (10\lambda_1))^2 = 4\lambda_1^2$$

If both $\lambda_1$ and $\lambda_2$ are large and $\lambda_1 = \lambda_2 = \lambda$ then we have:

$$R = \lambda_1 (\lambda_1) - \alpha(\lambda_1 + (\lambda_1))^2 = 0.95\lambda^2$$

Comparing the above two results we get $4\lambda_1^2$ where $\lambda_1$ is assumed to be relatively small value in the first case and $0.95\lambda^2$ where $\lambda$ is assumed to be a relatively large value in the second case. Therefore the second case would get a higher "corner-ness" score.

### Harris Corner Detector Algorithm

1. **Compute M matrix for every pixel:**
   $$M = \sum \begin{bmatrix} I_x^2 & I_x I_y \\ I_x I_y & I_y^2 \end{bmatrix}$$

2. **Compute eigenvalues** of every $M$ matrix

3. **Compute R** for every pixel using eigenvalues of corresponding $M$ matrices

4. **Threshold based on R-score** accordingly

5. **Return** top X corners

### Find Fundamental Matrix Algorithm

**Input:** $I_l, I_r$ (left and right images)

1. **Run RANSAC** for some number of iterations:
   - Select k matches from top A matches randomly
   - Compute Fundamental matrix by solving least squares problem: $\min \|M \mathbf{f}\|_2^2$
   - Store F matrix and error y

2. **Select best solution:** $\text{bestF} = \arg\min_y \text{Fmatrix}[F, y]$

3. **Return** bestF

---

*This blog post explains the mathematical foundation behind the Harris Corner Detector, a fundamental algorithm in computer vision for detecting corner features in images.*