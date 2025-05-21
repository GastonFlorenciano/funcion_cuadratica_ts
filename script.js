let modelo;

async function entrenarModelo() {
  document.getElementById("btnPredecir").disabled = true;

  modelo = tf.sequential();
  modelo.add(tf.layers.dense({ units: 8, inputShape: [2], activation: 'relu' }));
  modelo.add(tf.layers.dense({ units: 1 }));

  modelo.compile({
    optimizer: tf.train.adam(0.01),
    loss: 'meanSquaredError'
  });

  const xs = [];
  for (let i = -10; i <= 10; i += 0.5) {
    xs.push(i);
  }

  const inputFeatures = xs.map(x => [x, x * x]);
  const outputValues = xs.map(x => 2 * x * x - 3 * x + 1);

  const tensorXs = tf.tensor2d(inputFeatures);
  const tensorYs = tf.tensor2d(outputValues, [xs.length, 1]);

  await modelo.fit(tensorXs, tensorYs, {
    epochs: 300,
    batchSize: xs.length, 
    callbacks: {
      onTrainEnd: () => {
        document.getElementById("train-status").innerText = "✅ Entrenamiento finalizado. ¡Modelo listo!";
        document.getElementById("btnPredecir").disabled = false;
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
  const data = await prediccion.data();
  const y = data[0];

  if (isNaN(y) || !isFinite(y)) {
    document.getElementById("result").innerText = "Error en la predicción.";
    return;
  }

  document.getElementById("result").innerText = `Predicción: y = ${y.toFixed(2)}`;
}

entrenarModelo();
