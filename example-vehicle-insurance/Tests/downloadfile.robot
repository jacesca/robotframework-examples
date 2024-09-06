*** Settings ***
Library    OperatingSystem
Library    RequestsLibrary



*** Test Cases ***
Download Quote
    Download Insurance File


*** Keywords ***
Download Insurance File 
    [Arguments]    ${file_name}=Data/Quote-file.pdf    ${url}=http://sampleapp.tricentis.com/101/tcpdf/pdfs/quote.php
    Create Session    download    ${url}
    ${response} =    GET    ${url}
    Should Be Equal As Numbers     ${response.status_code}    200
    Create Binary File     ${file_name}     ${response.content}
    File Should Exist    ${file_name} 
