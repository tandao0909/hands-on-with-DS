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
