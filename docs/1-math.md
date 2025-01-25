# RANSOC: Real-time Adaptive Normalization for the Satisfaction of Curiosity

## Abstract

**RANSOC** (Real-time Adaptive Normalization for the Satisfaction of Curiosity) is a novel algorithm designed to balance relevance and exploration in search results. It achieves this by dynamically adjusting the weights of results based on user interactions, ensuring a broader exploration of the search space while maintaining result relevance. This approach is particularly beneficial in virtual environments, where users might otherwise limit their exploration to a subset of available options.

## 1. Introduction

Users often interact with systems that offer a wide range of potential outcomes, such as virtual characters with diverse facial expressions. However, they frequently explore only a limited subset of these possibilities, hindering their understanding of the system's full capabilities. RANSOC addresses this challenge by employing real-time adaptive normalization of search result weights, promoting the discovery of less prominent outcomes without sacrificing relevance to the user's query.

## 2. Problem Formulation

We model the search space as a set of $n$ planets, $P = \{p_1, p_2, ..., p_n\}$, each located in a $d$-dimensional space. A query is represented by a star $s$, also situated in this $d$-dimensional space. The objective is to rank planets based on their relevance to the star, while encouraging exploration by dynamically elevating the rank of previously overlooked planets.

### 2.1 Definitions

- **Planet Position:** The position of planet $p_i$ is given by:
  $p_i = \{p_{i1}, p_{i2}, ..., p_{id}\} \in \mathbb{R}^d$
- **Star Position:** The position of the query star $s$ is given by:
  $s = \{s_1, s_2, ..., s_d\} \in \mathbb{R}^d$
- **Planet Mass:** Each planet $p_i$ has a mass $m_i$, initially set to 1.

### 2.2 Distance Metric

The relevance of planet $p_i$ to star $s$ is quantified using the Euclidean distance:
$d(p_i, s) = \| p_i - s \|_2 = \sqrt{ \sum_{j=1}^{d} (p_{ij} - s_j)^2 }$

### 2.3 Weight Function

The initial weight of planet $p_i$ with respect to star $s$ is inversely proportional to the distance:
$w(p_i, s) = \frac{1}{d(p_i, s)}$

## 3. Adaptive Normalization

RANSOC introduces novelty by dynamically adapting planet masses over time, based on user interactions. This adjustment ensures that previously lower-ranked planets gain visibility.

### 3.1 Hit Detection

The planet with the highest weight is designated as the "hit" planet:
$p_{\text{hit}} = \arg\max_{p_i \in P} w(p_i, s)$

### 3.2 Mass Update Rule

After each hit, the planet masses are updated as follows:

- **Hit Planet:** The mass of the hit planet $p_{\text{hit}}$ is decreased:
  $m_{\text{hit}} = (1 - \alpha) m_{\text{hit}}$, where $\alpha$ is the learning rate ($0 < \alpha < 1$).
- **Other Planets:** The mass of all other planets $p_i$ (where $i \neq \text{hit}$) is increased:
  $m_i = \left(1 + \frac{\alpha}{n-1}\right) m_i$

### 3.3 Adaptive Weight

The updated weight of planet $p_i$ is computed using the modified mass and the distance:
$w_{\text{new}}(p_i, s) = \frac{m_i}{d(p_i, s)}$

## 4. RANSOC Algorithm

The RANSOC algorithm operates as follows:

1. **Initialization:** Set the mass $m_i$ of all planets $p_i \in P$ to 1.
2. **Weight Calculation:** For each planet $p_i$, calculate the weight $w(p_i, s)$.
3. **Hit Detection:** Identify the planet $p_{\text{hit}}$ with the maximum weight.
4. **Mass Update:** Update the masses of all planets using the mass update rule.
5. **Iteration:** Repeat steps 2-4 for each subsequent query.

### 4.1 Pseudocode

```python
def RANSOC(planets, star, alpha):
    """
    Real-time Adaptive Normalization for the Satisfaction of Curiosity.

    Args:
        planets: A list of planets, each represented as a d-dimensional vector.
        star: The query star, represented as a d-dimensional vector.
        alpha: The learning rate (0 < alpha < 1).

    Returns:
        The planet with the highest weight (the "hit" planet).
    """

    n = len(planets)
    masses = [1] * n  # Initialize masses to 1

    weights = [masses[i] / distance(planets[i], star) for i in range(n)]
    hit_index = weights.index(max(weights))

    masses[hit_index] *= (1 - alpha)
    for i in range(n):
        if i != hit_index:
            masses[i] *= (1 + alpha / (n - 1))
    
    return planets[hit_index]
```

## 5. Conclusion

RANSOC effectively balances relevance and novelty in search results by dynamically adapting weights based on user interactions. This encourages broader exploration of the search space, ensuring that even less relevant items gain visibility over time, thereby satisfying user curiosity.

## 6. Future Work

Future research will focus on integrating reinforcement learning techniques to further optimize user satisfaction by learning from user preferences over time.
