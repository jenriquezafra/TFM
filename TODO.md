## Short-term
- [X] Cambiar Brent por Levenberg-Marquardt
    - [X] implementar la función en 'implied_vol.py'
    - [X] cambiar en 'synth.yaml' para elección del rootfinder
    - [X] cambiar en 'gen_synth.py' la elección del rootfinder (cada uno tiene sus params)
    - [X] hacer tests


- [X] Generar nuevos datasets con todos los parámetros fijos menos 2 (uno de ellos gamma). Hacer LHS en d=2

- [X] Hacer un script para estudiar la sensibilidad (que genere plots también)

- [ ] Cambiar loss function a RMSE (root MSE)

- [ ] Training: en cada epoch (o cada 10), generar un nuevo batch para entrenar (re-sampling, no shuffle)
    - [ ] Verificar que eso no está ya implementado

- [ ] Probar la red en el nuevo datasets con params fijos y hacer surfaces plots par ver el error en esos 2 params (usar MSE/RMSE/MAE, ...)


## Medium-term


## Low priority
- [ ] En 'gen_synth.py', al fijar las variables, realmente estoy haciendo LHS sobre d=8, no sobre el que debería. No cambia mucho pero es conceptualmente incorrecto.

- [X] Poner de nuevo bien lo de la elección del preset en training (solo hay uno).

## Optionals
- [ ] Añadir L-BFGS como root finder