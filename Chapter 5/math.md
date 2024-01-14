# Convex set and convex function

## Introduction

- In previous chapters, our problem is actually an unconstrained optimization problem. Every ML problem is just an optimization problem (I talked about this in chapter 2).
- However, not only in ML, but in reality, optimization problems usually have many different constraints.
- In optimization, a constrained problem typically is written as:
    $$\underset{x}{minimize}f_0(x)$$
    subject to $g_i(x) \leq 0, i = 1, 2, \dots, m$ and $h_j(x)=0, i = 1, 2, \dots, n$
- In this expression:
    - The vector $\mathbf{x} = \left(x_1, \dots, x_n\right)^T$ is called the optimization variable.
    - The function $f_0(x): \mathbb{R}^n \rightarrow \mathbb{R}$ is the objective function. Examples of them are lost function in ML.
    - The functions $g_i, h_j: \mathbb{R}^n \rightarrow \mathbb{R}, i = 1, \dots, m, j = 1, \dots, n$ are called constraint functions, or simply constraints.
    - The set of all vector $\mathbf{x}$ satisfy the constraints named the feasible set.
    - A point in the feasible set is called a feasible point, otherwise is called infeasible set.
- Some note:
    - If the objective of the problem is to find the maximize, then you just need to change the sign of $f_0(\mathbf{x})$.
    - If the constraints are greater than or equal to (i.e. $g_i(\mathbf{x}) geq 0$), then you just need to change the sign of the constraints to obtain, for example, $-g_i(\mathbf{x})\leq 0$.
    - The constraint can be strictly greater than or strictly lower than.
    - If the constraint is in fact an equation, for example $h_i(\mathbf{x})=0$, you can rewritten it as two constraints $h_i(\mathbf{x}) \leq 0$ or $\mathbf{x} \geq 0$. Some books ignore an equation completely.
    - Optimization variables are typically written below the text `minimize`. When solving an optimization problem, remember to check which ones are variables, which ones are parameters.
- Optimization problems don't always have a general known way to solve them. A lot of them can't be solved optimally (yet), in fact.
- Most of known algorithms to find a solution usually find a local optimum, instead of a global optimum. In most cases, the local optimum is good enough for our purpose.
- In this specific note, we will talk about a small but important part of optimization field, named *convex optimization*, where the objective function is a convex function and the feasible set is a convex set. 

## Convex sets

- An *informal* definition of convex set:
    > A set is convex if the line connects any two points in that set is also belongs to that set.
- Some example of convex sets: A square, a circle excludes its border.
- Some example of nonconvex sets: The letter C, the Vietnam country.
- The mathematical, formal, rigorous definition of a convex set:
    > A set C is convex if for any given two points $x_1, x_2 \in C$, the point $x_\theta = \theta x_1 + (1-\theta)x_2$ is also belongs to C, for all $0 \leq \theta \leq 1$.
- You can see that $x_\theta = \theta x_1 + (1-\theta)x_2$ is just the line between $x_1$ and $x_2$.
- Under this definition, the empty set and the whole space are also considered convex sets. 

## Some important example of convex sets

- Hyperplane:
    > A hyperplane in n-dimension space is a set of all points $\mathbf{x} = (x_1, x_2, \dots, x_n)$ satisfy:
    $$a_1x_1+a_2x_2+\dots+a_nx_n = \mathbf{a}^T\mathbf{x} = b$$
    with $b, a_i, i = 1, 2, \dots, n$ are real numbers. <br>
    Prove: If $\mathbf{a}^T\mathbf{x_1} = b$ and $\mathbf{a}^T\mathbf{x_2}=b$, then $\mathbf{a}^T\mathbf{x_\theta}=\mathbf{a}^T\left(\theta \mathbf{x_1} + (1-\theta)\mathbf{x_2}\right)= \theta b + (1-\theta)b=b$.
- Halfspace:
    > A halfspace in n-dimension space is a set of all points $\mathbf{x} = (x_1, x_2, \dots, x_n)$ satisfy:
    $$a_1x_1+a_2x_2+\dots+a_nx_n = \mathbf{a}^T\mathbf{x} \leq b$$
    with $b, a_i, i = 1, 2, \dots, n$ are real numbers. <br>
    Prove: Similar to hyperplane.
- Norm closed balls:
    > Given a point c and a radius r, and the distance between any two points is already defined by a norm $\|.\|$. Then associated the norm closed ball is defined as:
    $$\overline{B}(c, r) = \{x | \|x-c\| \leq r\} = \{c + ru | \|u\| \leq 1\}$$
    Prove: Given the triangle inequality characteristic, we have:
    $$x_1, x_2 \in \overline{B}(c, r)$$ 
    $$\Rightarrow \|x_\theta-c\| = \|\theta x_1 + (1-\theta)x_2-c\| = \|\theta (x_1-c) + (1-\theta)(x_2-c)\| \leq \|\theta (x_1-c)\| + \|(1-\theta)(x_2-c)\| \leq \theta r + (1 - \theta) r = r.$$

## Intersection of convex sets

- Intersection of convex sets is a convex set.
- So the intersection of halfspaces and hyperplanes is also convex set.
- It is the convex polygon in 2-d space.
- In multi-dimensional space, it is called polyhedron (the plural form is polyhedra).
- Suppose there are m halfspace and n hyperplane. Every halfspace can be written as $\mathbf{a}_i^T\mathbf{x}\leq b_i, i = \overline{1, m}$ and every hyperplane can be written as $\mathbf{c}_i^T\mathbf{x} = d_i, i = \overline{1, n}$.
- So if we let $\mathbf(A) = (\mathbf{a}_1^T, \mathbf{a}_2^T, \dots, \mathbf{a}_m^T), b = (b_1, b_2, \dots, b_m)^T, \mathbf{C} = (\mathbf{c}_1^T, \mathbf{c}_2^T, \dots, \mathbf{c}_n^T), d = (d_1, d_2, \dots, d_n)$, then we can write the polyhedra as all the point $\mathbf{x}$ satisfy:
    $$\mathbf{A}^T\mathbf{x} = b,  \mathbf{C}^T\mathbf{x} = d$$

## Convex combination and Convex hull

- A point x is defined as a *convex combination* of $x_1, x_2, \dots x_k$ if it can be written as:
    $$ x = \theta_1x_1 + \theta_2x_2 + \dots + \theta_kx_k $$ 
    with $\theta_1+\theta_2+\dots+\theta_k=1$ and $\theta_i \geq 0, i=\overline{1, k}.$
- The convex hull of a set is defined as the set of all convex combination of that set.
- The convex hull of a set is a convex set.
- The convex hull of a convex set is itself.
- The convex hull of a set is the smallest convex superset of that set, i.e. it is the subset of every convex set consists of the original set.
- Now, we can define *linearly separable* in a mathematical way: Two sets are called *linearly separable* if their convex hull have no sharing point.
- So we have this theorem named **Separating hyperplane theorem**:
    > Two non-empty convex set C and D is nonintersecting if and only if there exists a vector $\mathbf{a}$ and a number satisfy:
    $$\mathbf{a}^T\mathbf{x} \leq b, \forall x \in C \text{ and } \mathbf{a}^T\mathbf{x} \geq b, \forall x \in D$$ 
- The set of all $\mathbf{x}$ satisfy $\mathbf{a}^T\mathbf{x} = b$ is a hyperplane. This hyperplane is called the separating hyperplane, also what we want to find in hard margin classification.

## Convex function

- First, we consider one variable function with its graph in a plane ($\mathbb{R}^2$).
- A function is convex if its domain is a convex set and if we connect a line between any two points in its graph lie above the graph or lie also on the graph.
- In other words, a function is convex if the line connects any two points on its graph *doesn't below* the graph.
- The domain set of a function usually called $\text{dom}f$.
- The mathematical definition if a convex function:
    > A function $f : \mathbb{R}^n \rightarrow \mathbb{R}$ is called a convex function if $\text{dom} f$ is a convex set and:
    $$f(\theta x  + (1-\theta) y) \leq \theta f(x) + (1-\theta)f(y), \forall x, y \in \text{dom}f, 0 \leq \theta \leq 1$$
- The condition $\text{dom}f$ is very important, as we need it to ensure $\theta x_1 + (1-\theta)x_2$ always in $\text{dom}f$, for all $\theta$, and therefore we can safely define $f(\theta x  + (1-\theta) y)$.
- A function f is called concave if -f is convex. 
- A function can be neither a convex or concave function.
- A linear function (which satisfy the Cauchy equation) is both convex an concave.
- The mathematical definition of a strictly convex function:
    > A function f is called *strictly convex* if $\text{dom} f$ is a convex set and 
    $$f(\theta x  + (1-\theta) y) < \theta f(x) + (1-\theta)f(y), \forall x, y \in \text{dom}f, 0 \leq \theta \leq 1$$
    It differentiates with a convex function at the smaller sign.
- Define similarly with *strictly concave* function.

## Basic properties

- If f(x) is convex, then af(x) is convex if a > 0 and af(x) is concave if a < 0. This is derived directly form the definition.
- The sum of two convex functions is a convex function, with the domain set is the intersection of the domain sets of two given functions (remain that the intersection of 2 convex sets is a convex set).
- If all function $f_1, f_2, \dots, f_m$ are convex, then their pointwise maximum function:
    $$f(x) = \max\{f_1(x), f_2(x), \dots, f_m(x)\}$$
    is also convex, with the domain set is the intersection of all the previous mentioned functions.
- Same for their *supremum* function.

## Examples

- The examples of convex function:
    - $y=ax+b$
    - $y=e^{ax}, a \in \mathbb{R}$
    - $y=x^a, x > 0, a \geq 1 \text{ or } a \leq 0$
    - The *negative entropy* function $y=x\log x, x > 0$
- The examples of concave function:
    - $y = ax + b$
    - $y = x^a, x>0, 0 \leq a \leq 1$
    - $y=\log(x), x>0$

## Affine functions

- All the functions $f(\textbf{x}) = \textbf{a}^T\textbf{x}+b$ are both convex and concave functions.
- If the variable is a matrix, then affine function is defined as:
    $$f(\textbf{X}) = \text{trace}\left(\textbf{A}^T\textbf{X}\right)+b$$
    where A is a matrix ensures that we can perform the multiplication and the result is a square matrix.

## Quadratic Form

- A quadratic single-variable has the form of $f(x) = ax^2+bx+c$ is convex if $a>0$, is concave if $a<0$.
- If the variable is a vector $\textbf{x} = (x_1, x_2, \dots, x_n)$, then the *quadratic form* is a function has a form of:
    $$f(\textbf{x}) = \textbf{x}^T\textbf{A}\textbf{x}+\textbf{b}^T\textbf{x}+c$$
    where $\textbf{A}, \textbf{B}$ are matrix and vector with suitable dimension and $\textbf{A}$ is usually a symmetric matrix.
- If $\textbf{A}$ is a positive semi-definite matrix, then $f(x)$ is a convex function. If $\textbf{A}$ is a negative semi-definite matrix, then $f(x)$ is a concave function.
- Recall that the loss function in linear regression has the form:
    $$\begin{align*}
    L(\textbf{w}) 
    &= \frac{1}{2m}\|\textbf{y}-\textbf{X}^T\textbf{w}\|^2 \\
    &= \frac{1}{2m}\left(\textbf{y}-\textbf{X}^T\textbf{w}\right)^T\left(\textbf{y}-\textbf{X}^T\textbf{w}\right) \\
    &= \frac{1}{2m} \textbf{w}^T\textbf{X}\textbf{X}^T\textbf{w} - \frac{1}{m}\textbf{y}^T\textbf{X}^T\textbf{w} + \frac{1}{2m}\textbf{y}^T\textbf{y}
    \end{align*}$$
- Because $\textbf{X}^T\textbf{X}$ is a positive semi-definite matrix, the loss function of linear regression is a convex function.

## Norms

- Every function satisfy three conditions of a norm is a convex function.
- Note that a norm function will always has a global minimum (the first condition of norm). Which means if we apply gradient descent on a norm function, we will almost surely obtain a very good result, if we set the learning rate schedule properly.

## Contours-level sets

- To investigate the convex property of hyperplane in 3D space, the method we usually use is *contour* (or *level set*).
- We define $\alpha$-sublevel set as follow:
    > $\alpha$-sublevel set of a function $f:\mathbb{R}^n \rightarrow \mathbb{R}$ is a set $\textbf{C}_\alpha$ defined as:
    $$\textbf{C}_\alpha=\{\textbf{x} \in \textbf{dom}f|f(x)\leq \alpha\}$$
- In other word, an $\alpha$-sublevel set of a function $f$ is the set of point x in the domain set of f such that $f(x) \leq \alpha$.
- Then we have this theorem:
    > If a function is convex, then all of its $\alpha$-sublevel set is convex. The opposite is not necessary true, i.e., if all of a function 's $\alpha$-sublevel set is a convex sets, that function may still not be a convex function.
- All the functions has a convex domain set and all of its $\alpha$-sublevel set is convex is called a *quasiconvex* function.
- The official definition of a quasiconvex function is:
    > A function $f: \textbf{C} \rightarrow \mathbb{R}$ with $\textbf{C}$ is a convex subset of $\mathbb{R}^n$ is called quasiconvex if for all $x, y \in \textbf{C}$ and $\theta \in [0, 1]$:
    $$f(\theta x + (1-\theta)y) \leq \max\{f(x), f(y)\}$$
- All convex function is quasiconvex, but the opposite is not true. For example, $f(x, y)=\frac{1}{10}(x^2+y^2-10\sin(\sqrt{x^2+y^2}))$ is quasiconvex, but not convex.

## Check if a function is convex using its derivate (and second-derivate)

- We can check if a function is convex using its first and second-order derivate. Of course, we can only do that if the function is differentiable in the first place.

### First-order condition

- Suppose a single-variable function f is differentiable at $x_0$, then the equation of the tangent line at $x_0$ is:
    $$y=f'(x_0)(x-x_0)+f(x_0)$$
- Suppose a multi-variable function f is differentiable at $\textbf{x}_0$, then the equation of the tangent hyperplane at $\textbf{x}_0$ is: 
    $$y=\nabla f(\textbf{x}_0)^T(\textbf{x}-\textbf{x}_0)+f(\textbf{x}_0)$$
- First-order condition:
    > Suppose a function f has a convex domain set, differentiable at every point in that domain set. Then f is convex if and only if for all $\textbf{x}, \textbf{x}_0$ in the domain set:
    $$f(\textbf{x}) \geq \nabla f(\textbf{x}_0)^T(\textbf{x}-\textbf{x}_0)+f(\textbf{x}_0)$$
- Similarly, a function is *strictly convex* if the equal sign happens if and only if $\textbf{x}=\textbf{x}_0$.
- Speak intuitively, a function is convex if and only if the tangent hyperplane at any point on the graph of the function does not above the graph.
- Example: If a symmetric $\textbf{A}$ is positive definite, then the function $f(\textbf{x}) = \textbf{x}^T\textbf{A}\textbf{x}$ is convex.
    > Prove: First-order derivate of $f(\textbf{x})$ is $\nabla f(\textbf{x})=2\textbf{A}\textbf{x}$. So the *first-order condition* can be written as (note that $\textbf{A}$ is a symmetric matrix):
    $$\textbf{x}^T\textbf{A}\textbf{x}\geq2(\textbf{A}\textbf{x}_0)^T(\textbf{x}-\textbf{x}_0)+\textbf{x}_0^T\textbf{A}\textbf{x}_0$$
    $$\Leftrightarrow \textbf{x}^T\textbf{A}\textbf{x}\geq2\textbf{x}_0^T\textbf{A}\textbf{x}-\textbf{x}_0\textbf{A}\textbf{x}_0$$
    $$(\textbf{x}-\textbf{x}_0)^T\textbf{A}(\textbf{x}-\textbf{x}_0)\geq 0$$
    The last inequality is correct, due to the fact that $\textbf{A}$ is a positive definite matrix. So $f(\textbf{x}) = \textbf{x}^T\textbf{A}\textbf{x}$ is convex.

### Second-order condition

- With multi-variable function, its second degree gradient is in fact a square symmetric matrix, named *Hessian*, annotated as $\nabla^2f(\textbf{x})$.
- Second-order condition:
    > A function f has a second-order gradient is convex if its domain set is convex and its Hessian matrix is a semi-positive definite matrix, for all x in the domain set:
    $$\nabla^2f(\textbf{x})\succeq 0$$
- If the Hessian matrix of a function is a positive definite matrix, then that function is strictly convex. Similarly, if its Hessian is a negative definite matrix, then that function is strictly concave.
- With single-variable $f(x)$, this condition is equivalent to $f''(x) \geq 0$, for all x in the domain set (and the domain set is convex).
- Some examples:
    - *Negative entropy* function $f(x)=x\log(x)$ is strictly convex, as its domain set $x>0$ is a convex set and $f''(x)=\frac{1}{x}>0, \forall x \in \textbf{dom}f$.
    - The function $f(x)=x^2+5\sin(x)$ is not a convex nor a concave function, as $f''(x)=2-5\sin(x)$ can have either negative or positive value.
    - The *cross entropy* function is strictly convex. For example, if you consider only two possibilities x and 1-x with a is a constant in [0, 1] and 0 < x < 1: $f(x) = -(a\log(x)+(1-a)\log(1-x))$ has second-order derivate is $f''(x)=\frac{a}{x^2}+\frac{1-a}{(1-x)^2}>0$.
    - 

# Convex optimization

- 