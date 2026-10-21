import React from 'react';
import { render } from 'react-dom';

import { marked } from 'marked';
import markedKatex from 'marked-katex-extension';
marked.use(markedKatex({ throwOnError: false }));

export default class Overview extends React.Component {
  constructor(props) {
    super(props);
  }
  render() {
    return (
      <div className="uk-section">
        {/* Intrinsic size + inline max-width keep the teaser within the viewport
            even before the stylesheet applies, and avoid layout shift. */}
        <img
          src={`${this.props.teaser}`}
          className="uk-align-center uk-responsive-width"
          width="3479"
          height="983"
          style={{ maxWidth: '100%', height: 'auto' }}
          alt=""
        />
        <h2 className="uk-text-bold uk-heading-line uk-text-center">
          <span>Abstract</span>
        </h2>
        <div
          lang="en"
          style={{ textAlign: 'justify', hyphens: 'auto' }}
          dangerouslySetInnerHTML={{
            __html: marked.parse(this.props.abstract),
          }}
        />
      </div>
    );
  }
}
