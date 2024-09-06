*** Settings ***
Library    OperatingSystem


*** Variables ***
&{MY-NEW-DICT}                username=demo    password=mode
&{MY-NEW-DICT2}                username=demo2    password=mode2


*** Keywords ***
Log My Username
    [Arguments]    ${USERNAME}
    Log            ${USERNAME}

Log My Password
    [Arguments]    ${PASSWORD}
    Log            ${PASSWORD}

Log Specific Username and Password
    [Arguments]        ${USERNAME}    ${PASSWORD}
    Log My Username    ${USERNAME}
    Log My Password    ${PASSWORD}
    Log                Final message
