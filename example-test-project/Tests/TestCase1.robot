*** Settings ***
Documentation    This is my project Test Case
Library    OperatingSystem

*** Keywords ***


*** Variables ***


*** Test Cases ***
TEST1
    [Tags]    Critical
    ${files_Count} =    Count Files In Directory    c:\\

TEST2
    [Tags]    Low
    Log    This is a sample test case.
