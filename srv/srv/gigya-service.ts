// Configuration for Gigya API
const GIGYA_API_KEY = process.env.GIGYA_API_KEY;
const GIGYA_USER_KEY = process.env.GIGYA_USER_KEY;
const GIGYA_SECRET_KEY = process.env.GIGYA_SECRET_KEY;
const GIGYA_ACCOUNTS_API_BASE_URL = process.env.GIGYA_ACCOUNTS_API_BASE_URL;

// Helper function for Gigya API calls
const callGigyaApi = async (endpoint: string, params: Record<string, any>) => {
  if (!GIGYA_API_KEY || !GIGYA_USER_KEY || !GIGYA_SECRET_KEY || !GIGYA_ACCOUNTS_API_BASE_URL) {
    throw new Error("Gigya API environment variables are not set.");
  }
  const url = `${GIGYA_ACCOUNTS_API_BASE_URL}.${endpoint}`;
  const allParams = {
    apiKey: GIGYA_API_KEY,
    userKey: GIGYA_USER_KEY,
    secret: GIGYA_SECRET_KEY,
    ...params
  };

  console.log(`Calling Gigya API: ${endpoint}`, JSON.stringify(allParams));

  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/x-www-form-urlencoded" },
    body: new URLSearchParams(allParams)
  });

  if (!response.ok) {
    const errorData = await response.text();
    console.error(`Gigya API Error (${response.status}): ${errorData}`);
    throw new Error(`Gigya API request failed for endpoint ${endpoint}`);
  }

  return response.json();
};

export interface GigyaUser {
  UID: string;
  profile?: {
    email?: string;
    firstName?: string;
    lastName?: string;
  };
  data?: {
    companyId?: string;
    company?: string;
    type?: string;
  };
  // Add other fields as necessary
}

export const searchUserInGigya = async (email: string): Promise<GigyaUser | null> => {
  const searchParams = {
    query: `SELECT UID, profile, data, preferences FROM accounts WHERE profile.email = '${email}'`
  };
  const searchData = await callGigyaApi("search", searchParams);
  if (searchData?.results?.length > 0) {
    return searchData.results[0] as GigyaUser;
  }
  return null;
};

export const updateUserInGigya = async (uid: string, profile: any, data: any) => {
  const updateParams = {
    UID: uid,
    profile: JSON.stringify(profile),
    data: JSON.stringify(data)
  };
  return callGigyaApi("setAccountInfo", updateParams);
};

export const insertUserInGigya = async (
  userData: {
    ID?: string;
    email?: string;
    firstName?: string;
    lastName?: string;
    companyId?: string;
    company?: string;
    type?: string;
  },
  preferences: Record<string, any>
) => {
  const profilePayload: Record<string, any> = { email: userData.email };
  if (userData.firstName) profilePayload.firstName = userData.firstName;
  if (userData.lastName) profilePayload.lastName = userData.lastName;

  const dataPayload: Record<string, any> = {};
  if (userData.companyId) dataPayload.companyId = userData.companyId;
  if (userData.company) dataPayload.company = userData.company;
  if (userData.type) dataPayload.type = userData.type;

  const insertParams: Record<string, any> = {
    isRegistered: true,
    isVerified: true,
    UID: userData.ID, // Consider using a Gigya generated UID if appropriate
    profile: JSON.stringify(profilePayload),
    ...(Object.keys(dataPayload).length > 0 && { data: JSON.stringify(dataPayload) }),
    preferences: JSON.stringify(preferences),
    lang: "en"
  };
  return callGigyaApi("importFullAccount", insertParams);
};
