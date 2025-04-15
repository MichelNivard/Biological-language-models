import numpy as np
import plotly.graph_objects as go
from scipy.spatial import procrustes
import sidechainnet as scn

def classical_mds(D):
    """Reconstruct 3D coords from distance matrix using classical MDS."""
    n = D.shape[0]
    D_squared = D ** 2
    H = np.eye(n) - np.ones((n, n)) / n
    B = -0.5 * H @ D_squared @ H
    eigvals, eigvecs = np.linalg.eigh(B)
    idx = np.argsort(eigvals)[::-1]
    eigvals = eigvals[idx]
    eigvecs = eigvecs[:, idx]
    L = np.diag(np.sqrt(np.maximum(eigvals[:3], 0)))
    V = eigvecs[:, :3]
    return V @ L

def show_plotly(coords, title="3D View", color=None):
    """Show interactive 3D Plotly plot."""
    x, y, z = coords[:, 0], coords[:, 1], coords[:, 2]
    color = color if color is not None else np.arange(len(coords))

    fig = go.Figure(
        data=[
            go.Scatter3d(
                x=x, y=y, z=z,
                mode='lines+markers',
                marker=dict(size=4, color=color, colorscale='Viridis'),
                line=dict(color='gray', width=2),
            )
        ]
    )
    fig.update_layout(
        title=title,
        scene=dict(
            xaxis_title='X',
            yaxis_title='Y',
            zaxis_title='Z',
            aspectmode='data'
        ),
        margin=dict(l=0, r=0, t=50, b=0),
    )
    fig.show()

def main():
    data = scn.load("debug", scn_dataset=True)
    for sample in data:
        coords = sample.coords
        if coords is None or coords.shape[0] < 10:
            continue
        ca_coords = coords[:, 0, :]
        if np.isnan(ca_coords).any():
            continue

        D = np.linalg.norm(ca_coords[:, None, :] - ca_coords[None, :, :], axis=-1)
        mds_coords = classical_mds(D)

        # Optional: scale MDS to match real protein size
        scale_real = np.linalg.norm(ca_coords)
        scale_mds = np.linalg.norm(mds_coords)
        mds_coords *= (scale_real / scale_mds)

        # Show raw MDS
        show_plotly(mds_coords, title="MDS Reconstruction (Before Alignment)")

        # Align using Procrustes (includes rotation/scale/translation)
        _, aligned_coords, _ = procrustes(ca_coords, mds_coords)

        # Show aligned version
        show_plotly(aligned_coords, title="MDS After Procrustes Alignment")

        # Optionally, also show the true structure
        show_plotly(ca_coords, title="True Protein Structure (Cα coords)")
        break

if __name__ == "__main__":
    main()
