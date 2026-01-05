"use client";

import paletteIcon from "@ui5/webcomponents-icons/dist/palette";
import {
  List,
  ListPropTypes,
  ResponsivePopover,
  ShellBar,
  ShellBarItem,
  ShellBarItemPropTypes,
  ListItemStandard,
  ButtonDomRef,
  ShellBarBranding,
  Avatar,
  Tag,
  Button,
  Label
} from "@ui5/webcomponents-react";
import { useRef, useState } from "react";
import classes from "./AppShellBar.module.css";
import { getTheme, setTheme } from "@ui5/webcomponents-base/dist/config/Theme";
import ListMode from "@ui5/webcomponents/dist/types/ListSelectionMode";
import { useNavigate } from "react-router-dom";
import "@ui5/webcomponents-icons/dist/menu2";

const THEMES = [
  { key: "sap_horizon", value: "Morning Horizon (Light)" },
  { key: "sap_horizon_dark", value: "Evening Horizon (Dark)" },
  { key: "sap_horizon_hcb", value: "Horizon High Contrast Black" },
  { key: "sap_horizon_hcw", value: "Horizon High Contrast White" }
];

export function AppShellBar() {
  const navigate = useNavigate();
  const popoverOpenerRef = useRef<ButtonDomRef | undefined>(undefined);
  const [popoverOpen, setPopoverOpen] = useState(false);
  const [currentTheme, setCurrentTheme] = useState(getTheme);

  const handleThemeSwitchItemClick: ShellBarItemPropTypes["onClick"] = (e) => {
    popoverOpenerRef.current = e.detail.targetRef as ButtonDomRef;
    setPopoverOpen(true);
  };
  const handleThemeSwitch: ListPropTypes["onSelectionChange"] = (e) => {
    const { targetItem } = e.detail;
    void setTheme(targetItem.dataset.key!);
    setCurrentTheme(targetItem.dataset.key!);
  };

  return (
    <>
      <ShellBar
        branding={
          <ShellBarBranding logo={<img alt="SAP Logo" src="/sap-logo-svg.svg" />} onClick={() => navigate("/")}>
            Utilities Tariff Mapping Cockpit
          </ShellBarBranding>
        }
        content={<Label>AI-assisted tariff PDF to IS-U configuration</Label>}
        profile={
          <Avatar>
            <img alt="person-placeholder" src="https://sap.github.io/ui5-webcomponents-react/v2/assets/Person-B7wHqdJw.png" />
          </Avatar>
        }
      >
        <ShellBarItem icon={paletteIcon} text="Change Theme" onClick={handleThemeSwitchItemClick} />
      </ShellBar>
      <ResponsivePopover
        className={classes.popover}
        open={popoverOpen}
        opener={popoverOpenerRef.current}
        onClose={() => {
          setPopoverOpen(false);
        }}
      >
        <List onSelectionChange={handleThemeSwitch} headerText="Change Theme" selectionMode={ListMode.Single}>
          {THEMES.map((theme) => (
            <ListItemStandard key={theme.key} selected={currentTheme === theme.key} data-key={theme.key} text={theme.value} />
          ))}
        </List>
      </ResponsivePopover>
    </>
  );
}

export default AppShellBar;
