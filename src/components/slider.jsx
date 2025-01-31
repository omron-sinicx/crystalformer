import React from 'react';
import { render } from 'react-dom';
import UIkit from 'uikit';

export default class Slider extends React.Component {
  componentDidMount() {
    this.UIkitComponent = UIkit.slider(this.gridElement, {
      center: true, // Center items mode
      autoplay: false, // Defines whether or not the slider items should switch automatically
    });
  }

  render() {
    return (
      <div
        className={this.props.className}
        ref={(element) => {
          this.gridElement = element;
        }}
        data-uk-slider
      >
        <div
          className="uk-position-relative uk-visible-toggle uk-light"
          tabIndex="-1"
        >
          {this.props.children}
        </div>
        <ul className="uk-slider-nav uk-dotnav uk-flex-center uk-margin"></ul>
      </div>
    );
  }
}
