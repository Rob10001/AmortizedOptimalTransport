# AmortizedOptimalTransport

**Goal:** Given two images A (source) and B (target), map the color palette of B onto A using entropic Optimal Transport (OT) solved by the Sinkhorn algorithm. Then, learn a neural network that predicts a warm-start to the Sinkhorn algorithm so that it can converge in fewer iterations.  

See Notebook "5. Demo" for visual examples.

## Helpful Readings

For additional context, [here](https://amsword.medium.com/a-simple-introduction-on-sinkhorn-distances-d01a4ef4f085) is a reading I found informative.

## Approach overview

### 1) Convert images into discrete color measures (color palettes)
For each of 2000 images, pixels were compressed into a palette of size `k` (e.g. `k = 128`) using k-means clustering. Distance is determined by Euclidean distance between two pixel colors in LAB color space. LAB distance is more perceptually similar to how humans perceive colors than RGB.
K-means yields:
*	Centroids: color centroids in LAB space representing the `k` most representative colors of the image
*	Centroid weights: normalized cluster masses (sum to 1)

### 2) Build a cost matrix for the Sinkhorn Algorithm
Compute pairwise color distances between palette centroids:
- `C_ij = d(A_i, B_j)`, where `d` is LAB distance

### 3) Compute entropic OT via Sinkhorn
Solve for the Sinkhorn transport plan:
- `P* = argmin_P  <P, C> + ε * KL(P || a b^T)` subject to `P 1 = a`, `P^T 1 = b`. The regularization term smooths the plan, yielding the Gibbs kernel `K = exp(-C / ε)` and the scaling form `P = diag(u) K diag(v)` for some `u` and `v`.

The Sinkhorn algorithm was solved by initializing `v` as the ones vector and then iteratively assigning `u = a/(Kv)` then `v=b/(K^Tu)` where we use elementwise division. Convergence in this context denotes when successive `log(v)` changes by less than `atol`.

**Numerical stability note:** naive exponentiation can underflow so implementations worked in the log domain instead

### 4) Train a neural net to predict a warm-start Sinkhorn transport plan `P0`
A small attention-based model that takes (centroids, weights) tuples `(C_A,W_A)` and `(C_B,W_B)` and outputs an initial plan `P0`. Using the 2000 input images, 100,000 image ordered pairs were split into an 80/10/10 training, validation, and testing split. The model was trained for 100 epochs. The loss vs. epochs graph can be seen within Notebook 4.

Some simplifications that were made:
- The NN outputs a row-stochastic matrix, which can be viewed as a conditional distribution over target colors per source color.
- Then additional Sinkhorn iterations were run to enforce both marginals if the relevant parameter (e.g. `use_as_warm_start` in `full_display_images` in Notebook 5) is true.


### 5) Evaluate: does warm-start reduce Sinkhorn iterations?
For held-out image pairs, I compare:
- baseline Sinkhorn iterations necessary until convergence
vs.
- Sinkhorn iterations necessary when initialized from the NN output (warm-start)

## Results
- The warm-start appeared to reduce iterations sometimes, but not entirely consistently and often negligibly. (Note that the number of iterations was capped at 20000).
- The five-number summary of the cold start iterations was (884, 2264, 3031, 4217, 20000).
- The five-number summary of the warm start iterations was (737, 2193, 2976, 4219, 20000).
- A histogram of all iterations is shown in Notebook 5.
- Future attempts could attempt to further fine-tune the neural network to achieve better results. Additionally, one could try to predict the `u` and `v` vectors in Sinkhorn instead of `P` while being careful of the fact that there exist infinitely many valid `u` and `v`.

---

## Repository structure

- `notebooks/`
  - **Loading Images into Drive**: extract a size 2000 sample of images from the Unsplash Lite dataset
  - **Loading Image Palettes into Drive**: extract palettes + weights from images, save to Drive
  - **Running Sinkhorn on Palettes**: compute OT plans in batches, handle numerical stability, save results
  - **Training/Testing NN**: train attention-based neural network to predict Sinkhorn warm-start plans
  - **Demo**: Plot iteration counts for warm start and standard start trials. Five examples that visually demonstrate the results of this project.
- `model_state_dict_k_128.pt`
  - trained model weights for a color palette of size `k=128`


## Running in Google Colab

1. Clone / open the notebooks in Colab
2. Mount Google Drive: This was my approach. Other approaches or paths used would require modification.
3. Run notebooks in ascending order.

---

## Key implementation notes

### Numerical stability
- Sinkhorn updates can underflow when `exp(-C/ε)` becomes tiny (common when `k` grows and/or `ε` is small).
- A log-domain Sinkhorn update avoids NaNs/zeros and makes batching feasible.

### Loss choice
- I treated transport plans as probability matrices due to their doubly-stochastic-esque nature and hence used KL-divergence-style losses.
- For activations, I preferred GELU over ReLU because features can be negative (LAB a, b channels) and we wish to avoid dying ReLUs.

---

## Next Steps

1. **Predict dual variables (`log u`/ `log v`)** instead of (or in addition to) `P0`  
   - This aligns more directly with how Sinkhorn converges and avoids having the model indirectly predict `K` which is already known.
2. **Use Sinkhorn within training**
   - Once NN predicts `P0`, run `T` Sinkhorn steps inside the training graph, and train on the resulting plan. This method of training would likely better align the model to produce efficient warm-starts.
3. **Rigorous hyperparameter tuning**
   - Thorough fine-tuning of Sinkhorn `epsilon`, Sinkhorn `atol`, `k` of k-means, `tol` of k-means, `source_enc` and `target_enc` layers in the NN, `hidden_dim` in NN, and other parameters was too expensive and time consuming for the scale of this project, hence end-to-end functionality and numerical stability were prioritized over exhaustive tuning. Fine-tuning could result in better warm-start performance.
