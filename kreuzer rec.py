import numpy as np

def kreuzer3F(hologram, z, L, wavelength, dx, deltaX, FC):
    """
    Reconstructs an in-line hologram using Kreuzer's method.

    Parameters:
    hologram (2D array): Hologram matrix
    z (float): Propagation distance
    L (float): Length parameter
    wavelength (float): Wavelength
    dx, deltaX (float): Pixel sizes at different stages
    FC (2D array): Cosine filter

    Returns:
    K (2D array): Reconstructed hologram
    """
    n_rows, n_cols = hologram.shape
    k = 2 * np.pi / wavelength
    W = dx * n_cols
    H = dx * n_rows

    deltaY = deltaX
    X, Y = np.meshgrid(np.arange(1, n_cols + 1), np.arange(1, n_rows + 1))

    xo = -W / 2
    yo = -H / 2
    xop = xo * L / np.sqrt(L * 2 + xo * 2)
    yop = yo * L / np.sqrt(L * 2 + yo * 2)

    deltaxp = xop / (-n_cols / 2)
    deltayp = yop / (-n_rows / 2)
    Xo = -deltaX * n_cols / 2
    Yo = -deltaY * n_rows / 2

    Xp = (dx * (X - n_cols / 2) * L) / np.sqrt(L * 2 + (dx * 2) * (X - n_cols / 2) * 2 + (dx * 2) * (Y - n_rows / 2) ** 2)
    Yp = (dx * (Y - n_rows / 2) * L) / np.sqrt(L * 2 + (dx * 2) * (Y - n_rows / 2) * 2 + (dx * 2) * (Y - n_rows / 2) ** 2)

    CHp_m = prepairholoF(hologram, xop, yop, Xp, Yp)



    Rp = np.sqrt(L * 2 - (deltaxp * X + xop) * 2 - (deltayp * Y + yop) ** 2)
    r = np.sqrt((deltaY*(2))(deltaX * (2)) * ((X - n_cols / 2) * 2 + (Y - n_rows / 2) * 2) + z * 2)
    CHp_m = ((L / Rp) * 4) * np.exp(-0.5j * k * (r * 2 - 2 * z * L) * Rp / (L * 2))



    pad = n_cols // 2

    # Redimensionar FC si no coincide
    if FC.shape != CHp_m.shape:
        FC = np.pad(FC, ((pad, pad), (pad, pad)), mode='constant')


    T1 = CHp_m * np.exp((1j * k / (2 * L)) * (2 * Xo * X * deltaxp + 2 * Yo * Y * deltayp + (X * 2) * deltaxp * deltaX + (Y * 2) * deltayp * deltaY))

    # Asegurarse de que la matriz T1 también se redimensione correctamente
    if T1.shape != FC.shape:
        T1 = np.pad(T1, ((pad, pad), (pad, pad)), mode='constant')

    K = propagate(T1, (L-z), wavelength, deltaX, deltaY)
    K = K[pad:pad + n_rows, pad:pad + n_rows]
    K = np.abs(K) ** 2
    K = normalize(K)

    return K