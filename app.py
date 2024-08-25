# app.py

from flask import Flask, render_template
import plotly.express as px
import plotly.io as pio

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/plot')
def plot():
    # Créer un graphique avec Plotly
    fig = px.line(x=[1, 2, 3, 4], y=[10, 20, 25, 30], title="Exemple de graphique interactif")
    graph_html = pio.to_html(fig, full_html=False)

    # Passer le code HTML du graphique au template
    return render_template('plot_plotly.html', graph_html=graph_html)

if __name__ == '__main__':
    app.run(debug=True)
