import React, { type ReactNode, useState } from "react";
import { useAuth } from "@site/src/authProviderBTP"; // Adjust path if necessary
import useBaseUrl from "@docusaurus/useBaseUrl";
import styles from "./styles.module.css";
import Link from "@docusaurus/Link";

interface LoginWallProps {
  children: ReactNode;
  message?: string;
  title?: string;
  renderOnlyWhenLoggedIn?: boolean;
  placement?: "overlay" | "below";
  showPreview?: boolean;
}

const LoginWall: React.FC<LoginWallProps> = ({
  children,
  message = "Log in or create a free SAP account to unlock this content. Get started in moments!",
  title = "Unlock Free Content",
  renderOnlyWhenLoggedIn = false,
  placement = "overlay",
  showPreview = false
}) => {
  const { isLoggedIn, login, isLoading, token } = useAuth();
  const logoUrl = useBaseUrl("/img/logo.svg");
  const privacyUrl = useBaseUrl("/privacy");
  const [termsAccepted, setTermsAccepted] = useState(false);
  const [showTermsError, setShowTermsError] = useState(false);

  const handleLogin = () => {
    // if (!termsAccepted) {
    //   setShowTermsError(true);
    //   return;
    // }

    setShowTermsError(false);
    login();
  };

  // Load the content during the "loading" state if there is a token (so that the table of contents links work)
  if (isLoggedIn || (token && isLoading)) {
    return <>{children}</>;
  }

  if (isLoading) {
    return (
      <div className={styles.loadingContainer}>
        <div className={styles.spinner}></div>
      </div>
    );
  }

  if (renderOnlyWhenLoggedIn) {
    return null;
  }

  // Show only the opening article blocks as a non-interactive preview.
  const preview = showPreview ? React.Children.toArray(children).filter(React.isValidElement).slice(0, 4) : null;

  return (
    <div className={`${styles.loginWallContainer}${placement === "below" ? ` ${styles.belowContent}` : ""}${showPreview ? ` ${styles.previewWall}` : ""}`}>
      {showPreview && <div className={styles.contentPreview} aria-hidden="true" inert>{preview}</div>}
      <div className={styles.loginWallOverlay}>
        <div className={styles.loginWallPopup}>
          <div className={styles.headerContainer}>Login or create a free SAP account</div>
          <div className={styles.contentContainer}>
            <h2>{title}</h2>
            <p>{message}</p>
          </div>
          {/* <div className={styles.divider}></div> */}
          <div className={styles.actionsWrapper} style={{ marginTop: "-10px", paddingTop: "0" }}>
            {/* <div className={styles.termsContainer}>
              <label className={`${styles.termsLabel} ${showTermsError ? styles.termsError : ""}`}>
                <input
                  type="checkbox"
                  checked={termsAccepted}
                  onChange={(e) => {
                    setTermsAccepted(e.target.checked);
                    if (e.target.checked) setShowTermsError(false);
                  }}
                  required
                  className={showTermsError ? styles.inputError : ""}
                />
                <span>
                  I have read and understand the Terms and Conditions of the SAP Business AI Platform Best Practices <span className={styles.required}>*</span>
                </span>
              </label>
              <p className={styles.termsText}>
                By clicking "Login with your SAP account", you agree to SAP Business AI Platform Best Practices' terms. You can learn about how SAP Business AI Platform Best Practices collects, uses, and
                shares data in the{" "}
                <Link to="https://btp-ai-bp.docs.sap/privacy" target="_blank" rel="noopener noreferrer">
                  SAP Business AI Platform Best Practices' Privacy Statement
                </Link>
                .
              </p>
            </div> */}

            <button onClick={handleLogin} className="button button--primary button--lg">
              <img src={logoUrl} alt="SAP Logo" /> Login with your SAP account
            </button>
            {showTermsError && <p className={styles.errorMessage}>You must accept the terms and conditions to continue.</p>}

            <p className={styles.termsText} style={{ marginBottom: "0", textAlign: "center" }}>
              By clicking "Login with your SAP account", you agree to SAP Business AI Platform Best Practices' terms. You can learn about how SAP Business AI Platform Best Practices collects, uses, and shares
              data in the{" "}
              <Link to={privacyUrl} target="_blank" rel="noopener noreferrer">
                SAP Business AI Platform Best Practices' Privacy Statement
              </Link>
              .
            </p>

            <div className={styles.orSeparator}>or</div>
            <Link to="https://account.sap.com/register" target="_blank" className={styles.createAccountLink}>
              Create your free SAP account
            </Link>
          </div>
        </div>
      </div>
    </div>
  );
};

export default LoginWall;
