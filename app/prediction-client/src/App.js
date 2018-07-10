import React, { Component } from 'react';
import axios from 'axios';
import logo from './logo.svg';
import './App.css';

class App extends Component {
	state = {
		predictions: []
	}

	componentDidMount() {
		axios.get('http://localhost:5000/predict')
			.then(res => {
				const predictions = res.data;
				this.setState({predictions});
			});
	}
  render() {
    return (
      <div className="App">
        <header className="App-header">
          <img src={logo} className="App-logo" alt="logo" />
          <h1 className="App-title">Welcome to React</h1>
        </header>
       	<ul>
					{ this.state.predictions.map(prediction => <li>{prediction.class} -- {prediction.score}</li>) } 
				</ul> 
      </div>
    );
  }
}

export default App;
