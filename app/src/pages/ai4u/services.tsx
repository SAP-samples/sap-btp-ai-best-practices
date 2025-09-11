import React, { useEffect } from "react";
import Head from "@docusaurus/Head";

export default function AI4UServicesRedirect(): JSX.Element {
  const redirectUrl = "https://ai4u-website.cfapps.eu10-004.hana.ondemand.com/services";

  useEffect(() => {
    // Redirect immediately when component mounts
    window.location.href = redirectUrl;
  }, []);

  return (
    <>
      <Head>
        <meta httpEquiv="refresh" content={`0; url=${redirectUrl}`} />
        <title>Redirecting to AI4U Services...</title>
      </Head>
      <div
        style={{
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          justifyContent: "center",
          minHeight: "50vh",
          textAlign: "center",
          padding: "2rem"
        }}
      >
        <h1>Redirecting to AI4U Services...</h1>
        <p>
          If you are not redirected automatically, please <a href={redirectUrl}>click here</a>.
        </p>
      </div>
    </>
  );
}
