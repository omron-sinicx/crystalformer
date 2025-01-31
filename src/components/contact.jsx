import React from 'react';
import { render } from 'react-dom';
import { FaGithub } from 'react-icons/fa6';
import { MdAlternateEmail, MdContactMail } from 'react-icons/md';

class ContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <MdContactMail size="3em" />
        </div>
        <div className="uk-width-expand">
          <span className="uk-comment-title uk-margin-remove">
            <a
              target="_blank"
              className="uk-link-reset"
              href={this.props.author.url}
            >
              {this.props.author.name}
            </a>
          </span>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li className="uk-visible@m">
              <a href="#">{this.props.author.position}</a>
            </li>
            <li>
              <a href="https://www.omron.com/sinicx/" target="_blank">
                OMRON SINIC X
              </a>
            </li>
          </ul>
        </div>
      </div>
    );
  }
}

class OmronContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <MdAlternateEmail size="3em" />
        </div>
        <div className="uk-width-expand">
          <span className="uk-comment-title uk-margin-remove">
            <a className="uk-link-reset">contact@sinicx.com</a>
          </span>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li>
              <a href="https://www.omron.com/sinicx/" target="_blank">
                OMRON SINIC X
              </a>
            </li>
          </ul>
        </div>
      </div>
    );
  }
}

class GithubContactCard extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-width-1-2@s uk-flex">
        <div className="uk-width-auto uk-margin-right">
          <FaGithub size="3em" />
        </div>
        <div className="uk-width-expand">
          <span className="uk-comment-title uk-margin-remove">
            <a
              className="uk-link-reset"
              target="_blank"
              href={this.props.issues}
            >
              GitHub issues
            </a>
          </span>
          <ul className="uk-comment-meta uk-subnav uk-subnav-divider uk-margin-remove-top">
            <li>
              <a href={this.props.repo} target="_blank">
                GitHub.com
              </a>
            </li>
          </ul>
        </div>
      </div>
    );
  }
}
export default class Contact extends React.Component {
  constructor(props) {
    super(props);
  }

  render() {
    return (
      <div className="uk-section">
        <h2 className="uk-heading-line uk-text-center">Contact</h2>
        <div className="uk-grid-medium" data-uk-grid>
          {this.props.contact_ids.map((cid) => {
            if (cid == 'omron') {
              return <OmronContactCard key={'contact-omron'} />;
            } else if (cid == 'github') {
              return (
                <GithubContactCard
                  repo={this.props.resources.code}
                  issues={this.props.resources.code + '/issues'}
                  key={'contact-github'}
                />
              );
            } else {
              return (
                <ContactCard
                  author={this.props.authors[cid - 1]}
                  key={'contact-' + cid - 1}
                />
              );
            }
          })}
        </div>
      </div>
    );
  }
}
