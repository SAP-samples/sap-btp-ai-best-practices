import React, { useEffect } from "react";
import Head from "@docusaurus/Head";

export default function DataScientistRedirect(): JSX.Element {
  const redirectUrl =
    "https://ai4u-website.cfapps.eu10-004.hana.ondemand.com/data-scientist";

  useEffect(() => {
    window.location.href = redirectUrl;
  }, []);

  return (
    <>
      <Head>
        <meta httpEquiv="refresh" content={`0; url=${redirectUrl}`} />
        <title>Redirecting to Data Scientist...</title>
      </Head>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "50vh",
          textAlign: "center",
          padding: "2rem",
        }}
      >
        <h1>Redirecting to Data Scientist...</h1>
        <p>
          If you are not redirected automatically, please{" "}
          <a href={redirectUrl}>click here</a>.
        </p>
      </div>
    </>
  );
}
