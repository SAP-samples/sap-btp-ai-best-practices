import { environment } from "@site/src/config/environment";

export const linkEmailToTrackingData = ({ email }: { email: string }) => {
  const storedTrackingInformationEncoded = localStorage.getItem(environment.storageName);
  if (!storedTrackingInformationEncoded) {
    return;
  }

  const anonymousEmailEncoded = localStorage.getItem(`${environment.storageName}-anonymousEmail`);
  const anonymousEmail = anonymousEmailEncoded ? atob(anonymousEmailEncoded) : null;

  try {
    const storedTrackingInformation = JSON.parse(atob(storedTrackingInformationEncoded));

    // If there's no anonymousEmail, it means the email is not linked
    if (!anonymousEmail) {
      localStorage.setItem(`${environment.storageName}-anonymousEmail`, btoa(storedTrackingInformation.email));
    }

    storedTrackingInformation.email = email;

    localStorage.setItem(environment.storageName, btoa(JSON.stringify(storedTrackingInformation)));
  } catch (error) {
    console.error("Failed to parse tracking information from localStorage:", error);
    localStorage.removeItem(environment.storageName);
    localStorage.removeItem(`${environment.storageName}-anonymousEmail`);
  }
};

export const unlinkEmailFromTrackingData = () => {
  const storedTrackingInformationEncoded = localStorage.getItem(environment.storageName);
  if (!storedTrackingInformationEncoded) {
    return;
  }

  const anonymousEmailEncoded = localStorage.getItem(`${environment.storageName}-anonymousEmail`);
  const anonymousEmail = anonymousEmailEncoded ? atob(anonymousEmailEncoded) : null;

  try {
    const storedTrackingInformation = JSON.parse(atob(storedTrackingInformationEncoded));

    // If there's no anonymousEmail, there is nothing to unlink
    if (!anonymousEmail) {
      return;
    }

    storedTrackingInformation.email = anonymousEmail;
    localStorage.removeItem(`${environment.storageName}-anonymousEmail`);

    localStorage.setItem(environment.storageName, btoa(JSON.stringify(storedTrackingInformation)));
  } catch (error) {
    console.error("Failed to parse tracking information from localStorage:", error);
    localStorage.removeItem(environment.storageName);
    localStorage.removeItem(`${environment.storageName}-anonymousEmail`);
  }
};
