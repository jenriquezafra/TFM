## Short-term
- [X] Cambiar Brent por Levenberg-Marquardt
    - [X] implementar la función en 'implied_vol.py'
    - [X] cambiar en 'synth.yaml' para elección del rootfinder
    - [X] cambiar en 'gen_synth.py' la elección del rootfinder (cada uno tiene sus params)
    - [X] hacer tests


- [X] Generar nuevos datasets con todos los parámetros fijos menos 2 (uno de ellos gamma). Hacer LHS en d=2

- [X] Hacer un script para estudiar la sensibilidad (que genere plots también)

- [X] Cambiar loss function a RMSE (root MSE)

- [X] Training: en cada epoch (o cada 10), generar un nuevo batch para entrenar (re-sampling, no shuffle)
    - [X] Verificar que eso no está ya implementado
    - (Ya estaba implementado)

- [X] Probar la red en el nuevo datasets con params fijos y hacer surfaces plots par ver el error en esos 2 params (usar MSE/RMSE/MAE, ...)
    - [X] Cambiar un poco 

- [X] Sacar los heatmaps de kappa que faltan

- [X] Añadir L-BFGS optimizer
    - [X] Mix con Adam cada x epochs

- [ ] Hacer ramas en GitHub: una para optimizadores diferentes (ADAM, L-BFGS y mix). Usar mismos datos.
    - [ ] Hacer runs IGUALES con cada optimizador.
## Medium-term


## Low priority
- [ ] En 'gen_synth.py', al fijar las variables, realmente estoy haciendo LHS sobre d=8, no sobre el que debería. No cambia mucho pero es conceptualmente incorrecto.

- [X] Poner de nuevo bien lo de la elección del preset en training (solo hay uno).

- [X] Arreglar en 'sensitivity_pricer.py' lo de que hacer raices negativas (ejecutar y ver el log)


## Optionals
- [X] Añadir L-BFGS como root finder

- [X] Añadir mix de L-BFGS y ADAM
