let modelo;

async function entrenarModelo() {
  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 1, inputShape: [2] }));

  modelo.compile({
    optimizer: tf.train.sgd(0.01),
    loss: 'meanSquaredError'
  });

  const xs = [-6, -5, -4, -3, -2, -1, 0, 1, 2];

  const inputFeatures = xs.map(x => [x, x * x]);
  const outputValues = xs.map(x => 2 * x * x - 3 * x + 1);

  const tensorXs = tf.tensor2d(inputFeatures);
  const tensorYs = tf.tensor2d(outputValues, [xs.length, 1]);

  await modelo.fit(tensorXs, tensorYs, {
    epochs: 500,
    callbacks: {
      onTrainEnd: () => {
        document.getElementById("train-status").innerText = "✅ Entrenamiento finalizado. ¡Modelo listo!";
      }
    }
  });
}

async function hacerPrediccion() {
  const x = parseFloat(document.getElementById("inputX").value);
  
  if (isNaN(x)) {
    document.getElementById("result").innerText = "Por favor, ingresa un número válido.";
    return;
  }

  const inputTensor = tf.tensor2d([[x, x * x]]);
  const prediccion = modelo.predict(inputTensor);
  const y = (await prediccion.data())[0];

  document.getElementById("result").innerText = `Predicción: y = ${y.toFixed(2)}`;
}

entrenarModelo();
