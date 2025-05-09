import { useLocation, useHistory } from "@docusaurus/router";
import React, { createContext, useState, useContext, useEffect, useMemo } from "react";
import siteConfig from "@generated/docusaurus.config";
import { unlinkEmailFromTrackingData, linkEmailToTrackingData } from "@site/src/lib/trackingTool/trackingStorage";

const BTP_API = siteConfig.customFields.apiUrl as string;

interface AuthContextProps {
  isLoggedIn: boolean;
  login: () => Promise<void>;
  logout: () => void;
  user: UserInfo | null;
  isLoading: boolean;
}

interface UserInfo {
  ID: string;
  firstName: string;
  lastName: string;
  email: string;
  company: string;
  companyId: string[];
  type: string;
}

const AuthContext = createContext<AuthContextProps>({
  isLoggedIn: false,
  login: async () => {},
  logout: () => {},
  user: null,
  isLoading: true
});

interface AuthProviderProps {
  children: React.ReactNode;
}

const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [user, setUser] = useState<UserInfo | null>(null);
  const [token, setToken] = useState("");
  const [isLoading, setIsLoading] = useState(true);
  const location = useLocation();
  const history = useHistory();

  const login = async () => {
    // Redirect to login
    window.location.href = `${BTP_API}/user/login?origin_uri=${window.location.href}`;
  };

  const logout = () => {
    unlinkEmailFromTrackingData();
    setIsLoggedIn(false);
    setUser(null);
    setToken("");
    localStorage.removeItem("token");
  };

  const value = useMemo(
    () => ({
      isLoggedIn,
      login,
      logout,
      user,
      isLoading
    }),
    [isLoggedIn, login, logout, user, isLoading]
  );

  useEffect(() => {
    if (token > "") {
      localStorage.setItem("token", token);
    } else if (localStorage.getItem("token")) {
      setToken(localStorage.getItem("token"));
    }
  }, [token]);

  useEffect(() => {
    const checkAuth = async () => {
      setIsLoading(true);
      const params = new URLSearchParams(location.search);
      const queryToken = params.get("t");
      let currentToken = localStorage.getItem("token");

      if (queryToken) {
        setToken(queryToken);
        localStorage.setItem("token", queryToken);
        currentToken = queryToken;
        history.replace(location.pathname);
      } else if (currentToken) {
        setToken(currentToken);
      } else {
        setIsLoading(false);
        return;
      }

      // console.log(`Checking auth with token: ${currentToken}`);

      if (currentToken) {
        try {
          const responseUser = await fetch(`${BTP_API}/user/getUserInfo`, {
            headers: {
              Authorization: `Bearer ${currentToken}`
            },
            mode: "cors"
          });
          // console.log(responseUser);
          if (responseUser.ok) {
            const dataUser = await responseUser.json();
            // console.log(dataUser);
            setUser(dataUser);
            setIsLoggedIn(true);

            // Update Tracking Tool User
            linkEmailToTrackingData({ email: dataUser.email });
          } else {
            console.error("Failed to fetch user info, potentially invalid token");
            logout();
          }
        } catch (error) {
          console.error("Error during auth check:", error);
          logout();
        }
      } else {
        unlinkEmailFromTrackingData();
      }
      setIsLoading(false);
    };

    checkAuth();
  }, [location, history]);

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

const useAuth = () => {
  return useContext(AuthContext);
};

export { AuthProvider, useAuth };
