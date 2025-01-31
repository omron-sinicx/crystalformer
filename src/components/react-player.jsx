import React from 'react';
import ReactPlayer from 'react-player/lazy';
import Slider from '../components/slider.jsx';
import { render } from 'react-dom';

export default class Player extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return this.props.demo ? (
      <div className="uk-section">
        <h3 className="uk-heading-line uk-text-center">DEMO</h3>
        <Slider>
          <ul className="uk-slider-items">
            {this.props.demo.map((d, idx) => {
              return (
                <li className="uk-panel" key={'mp4-' + idx}>
                  <ReactPlayer
                    className="react-player uk-align-center"
                    url={require('../videos/' + d.mp4)}
                    width={d.scale}
                    height={d.scale}
                    playing={true}
                    loop={true}
                    muted={true}
                    controls={false} // show native controller
                  />
                  <div className="uk-overlay uk-overlay-primary uk-position-bottom">
                    <p>{d.text}</p>
                  </div>
                </li>
              );
            })}
          </ul>
        </Slider>
      </div>
    ) : null;
  }
}
