# -*- coding: utf-8 -*-
import argparse, os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

# GAN simple (no GP): funciona bien en vectors N<~100 con BN/Dropout y label smoothing
def build_generator(z_dim, n_out):
    model = models.Sequential(name='G')
    model.add(layers.Input(shape=(z_dim,)))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.BatchNormalization())
    model.add(layers.Dense(n_out))  # salida z-score (sin activación)
    return model

def build_discriminator(n_in):
    model = models.Sequential(name='D')
    model.add(layers.Input(shape=(n_in,)))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(256))
    model.add(layers.LeakyReLU(0.2))
    model.add(layers.Dropout(0.3))
    model.add(layers.Dense(1, activation='sigmoid'))
    return model

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--x-npy', required=True)           # T x N (z-score, salida de 4a)
    ap.add_argument('--z-dim', type=int, default=64)
    ap.add_argument('--epochs', type=int, default=4000)
    ap.add_argument('--batch', type=int, default=64)
    ap.add_argument('--out-gen', required=True)         # ruta .keras del generador
    args = ap.parse_args()

    X = np.load(args.x_npy).astype('float32')           # (T, N)
    N = X.shape[1]
    z_dim = args.z_dim

    G = build_generator(z_dim, N)
    D = build_discriminator(N)
    D.compile(optimizer=optimizers.Adam(1e-4, beta_1=0.5), loss='binary_crossentropy')

    z = layers.Input(shape=(z_dim,))
    fake = G(z)
    D.trainable = False
    validity = D(fake)
    GAN = models.Model(z, validity)
    GAN.compile(optimizer=optimizers.Adam(1e-4, beta_1=0.5), loss='binary_crossentropy')

    # Entrenamiento
    valid = np.ones((args.batch, 1), dtype='float32') * 0.9   # label smoothing
    fake_ = np.zeros((args.batch, 1), dtype='float32')

    steps = args.epochs
    rng = np.random.default_rng(1234)
    for step in range(1, steps+1):
        # 1) batch real
        idx = rng.integers(0, X.shape[0], size=args.batch)
        real_batch = X[idx]

        # 2) batch fake
        z_samp = rng.normal(size=(args.batch, z_dim)).astype('float32')
        gen_batch = G.predict(z_samp, verbose=0)

        # 3) entrenar D
        d_loss_real = D.train_on_batch(real_batch, valid)
        d_loss_fake = D.train_on_batch(gen_batch,  fake_)
        d_loss = 0.5 * (d_loss_real + d_loss_fake)

        # 4) entrenar G
        z_samp = rng.normal(size=(args.batch, z_dim)).astype('float32')
        g_loss = GAN.train_on_batch(z_samp, valid)

        if step % 200 == 0:
            print(f"[{step}/{steps}] D_loss={d_loss:.4f}  G_loss={g_loss:.4f}")

    os.makedirs(os.path.dirname(args.out_gen), exist_ok=True)
    G.save(args.out_gen)
    print("[OK] Generador guardado →", args.out_gen)

if __name__ == '__main__':
    main()
