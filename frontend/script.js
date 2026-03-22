async function predict() {
  console.log("🔥 Button clicked");

  const data = {
    tenure: Number(document.getElementById("tenure").value) || 0,
    MonthlyCharges: Number(document.getElementById("MonthlyCharges").value) || 0,
    gender: document.getElementById("gender").value
  };

  try {
    const res = await fetch("/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify(data)
    });

    const result = await res.json();

    document.getElementById("result").innerText =
      result.prediction == 1 ? "❌ Will Churn" : "✅ Will Stay";

  } catch (err) {
    console.error(err);
    document.getElementById("result").innerText = "❌ Error";
  }
}