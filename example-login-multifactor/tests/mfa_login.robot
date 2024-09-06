*** Settings ***
Library    Browser
Library    ../Resources/totp.py
Suite Setup    New Browser    browser=${BROWSER}    headless=${HEADLESS}   # performed once for entire Suite (all tests)  # noqa
Test Setup    New Context   # performed before every test
Test Teardown    Close Context
Suite Teardown    Browser.Close Browser


*** Variables ***
${BROWSER}    chromium
${HEADLESS}    False


*** Test Cases ***
Login with MFA
    New Page    https://seleniumbase.io/realworld/login
    Fill Text    id=username    demo_user
    Fill Text    id=password    secret_pass
    ${totp}    Get Totp    GAXG2MTEOR3DMMDG
    Fill Text    id=totpcode     ${totp}
    Take Screenshot    mfa-login.png
    Sleep    3s
    Click    "Sign in"
    Get Text  h1  ==  Welcome!
    Sleep    3s
