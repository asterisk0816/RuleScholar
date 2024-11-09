function doGet() {
  // var now = new Date();
  // var hours = now.getHours();

  // if (hours >= 8 && hours < 16) {
  //   return HtmlService.createTemplateFromFile('opened').evaluate()
  //     .append('<script>var loginTimerStart = ' + new Date().getTime() + ';</script>');
  // } else {
  //   return HtmlService.createTemplateFromFile('closed').evaluate();
  // }

  //検証用（通常時はコメントアウトすること）
  return HtmlService.createTemplateFromFile('opened').evaluate()
    .append('<script>var loginTimerStart = ' + new Date().getTime() + ';</script>');
}

function include(filename) {
  return HtmlService.createHtmlOutputFromFile(filename)
    .getContent();
}

function recordExecution() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('アクセス制限検出');
  var timestamp = new Date();
  sheet.appendRow([timestamp]);
}

// メールアドレスの形式と一意性を検証する関数
function validateAndRecordEmail(enteredEmail, password) {
  Logger.log("validateAndRecordEmail called with: " + enteredEmail);
  var email = enteredEmail + "@example.com";
  Logger.log("Generated email: " + email);
  var startTime = new Date(); // 関数の開始時刻を記録

  // 範囲指定検証
  var emailNumber = parseInt(enteredEmail, 10);
  if (emailNumber !== 0 && (emailNumber < 32181001 || emailNumber > 32246999)) {
    Logger.log("Email out of range: " + enteredEmail);
    return {status: 'out_of_range'};
  }

  if (new Date() - startTime > 60000) { // 現在時刻が開始時刻から1分以上経過しているかチェック
    Logger.log("Function timed out");
    return {status: 'timeout'};
  }

  // メールアドレス記録シートの51行目にデータがあるかチェック
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('メールアドレス記録（毎日リセット）');
  var row101 = sheet.getRange(51, 1).getValue();
  if (row101) {
    Logger.log("51行目にデータが存在: 処理中止");
    return {status: 'row_51_filled'};
  }

  if (new Date() - startTime > 60000) { // 現在時刻が開始時刻から1分以上経過しているかチェック
    Logger.log("Function timed out");
    return {status: 'timeout'};
  }

  // 特殊なメールアドレス「00000000」のチェック
  if (enteredEmail === '00000000') {
    Logger.log("Admin login detected: " + email);
    // 「ログイン情報（管理者）」シートから1行目のデータを取得
    var adminSheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('ログイン情報（管理者）');
    var adminData = adminSheet.getRange(1, 1).getValue();
    var adminEmail = adminData.split('（')[0];
    var adminPassword = adminData.match(/\（([^\）]+)\）/)[1];

    if (adminEmail === email && adminPassword === password) {
      Logger.log("Admin login successful: " + email);
      return {status: 'ok', email: email};
    } else {
      Logger.log("Invalid admin credentials: " + email);
      return {status: 'login_failed'};
    }
  }

  // メールアドレスから年度と行番号を取得
  var yearCode = emailNumber === 0 ? '00' : enteredEmail.substring(2, 4);
  var rowNumber = parseInt(enteredEmail.substring(4)) - 1000; // 末尾4桁から1000
  var loginSheetName = 'ログイン情報（' + yearCode + '）';
  var loginSheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName(loginSheetName);

  var loginMatch = false;

  if (rowNumber <= 0 || rowNumber > 5999) {
    Logger.log("Invalid row number: " + rowNumber);
    return {status: 'invalid_row_number'};
  }

  // 対応する行のデータを取得
  var rowData = loginSheet.getRange(rowNumber, 1).getValue();
  var matchedEmail = rowData.split('（')[0];
  var matchedPassword = rowData.match(/\（([^\）]+)\）/)[1];

  if (matchedEmail === email && matchedPassword === password) {
    loginMatch = true;
  }

  if (!loginMatch) {
    Logger.log("Invalid login credentials for email: " + email);
    return {status: 'login_failed'};
  }

  if (new Date() - startTime > 60000) { // 現在時刻が開始時刻から1分以上経過しているかチェック
    Logger.log("Function timed out");
    return {status: 'timeout'};
  }

  // メールアドレスの一意性を検証
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('メールアドレス記録（毎日リセット）');
  var data = sheet.getDataRange().getValues();
  for (var i = 1; i < data.length; i++) {
    if (data[i][0] === email) {
      Logger.log("Duplicate email found: " + email);
      return {status: 'duplicate'};
    }
  }
  
  // メールアドレスが00000000の場合は記録しない
  if (email !== '00000000@example.com') {
    // メールアドレスを記録
    sheet.appendRow([email]);
    Logger.log("Email recorded: " + email);
  } else {
    Logger.log("Email not recorded (00000000): " + email);
  }

  if (new Date() - startTime > 60000) { // 現在時刻が開始時刻から1分以上経過しているかチェック
    Logger.log("Function timed out");
    return {status: 'timeout'};
  }

  recordExecution();
  Logger.log("Validation successful for email: " + email);
  return {status: 'ok', email: email};
};

// ChatGPTの関数
function chatGPT(email, talk_str) {

  var now = new Date();
  var hours = now.getHours();

  // 検証用常時利用不可
  return "【システム通知】検証用のため、ChatGPT本体を利用することはできません。ログイン画面など、他の部分でエラーが発生しないか、ご確認をお願い致します。";

  // 指定された時間外なら特定のメッセージを返す
  if (hours < 8 || hours >= 17) {
    return "【システム通知】只今の時間は休止中です。後日お問い合わせください。";
  }

  // 不適切な言葉のリスト
  var inappropriateWords = ['不適切な単語1', '不適切な単語2', '不適切な単語3'];

  // 会話内容に不適切な言葉が含まれているかチェック
  for (var i = 0; i < inappropriateWords.length; i++) {
    if (talk_str.includes(inappropriateWords[i])) {
      return "【システム通知】申し訳ありませんが、その内容には回答できません。";
    }
  }

  if (talk_str) {
    const requestBody = {
      "model": "gpt-3.5-turbo",
      "messages": [{'role': 'system', 'content': "あなたは優秀なアシスタントです。学校のルールについて正確に回答できます。学校のルールに関する質問以外（＝学校以外の内容）には「絶対に」回答しないでください。"},{'role': 'user', 'content': talk_str}],
      "temperature": 1.0,
      "max_tokens": 100
    };

    const requestOptions = {
      "method": "POST",
      "headers": {
        "Content-Type": "application/json",
        "Authorization": "Bearer " + PropertiesService.getScriptProperties().getProperty('APIKEY')
      },
      "payload": JSON.stringify(requestBody)
    };

    var response = JSON.parse(UrlFetchApp.fetch("https://api.openai.com/v1/chat/completions", requestOptions).getContentText());
    var answer = response.choices[0].message.content.trim();

    // 元の回答を保存する変数を追加
    var originalAnswer = answer;

    // ChatGPTの応答に対する不適切な言葉のリスト
    var responseInappropriateWords = ['不適切な単語A', '不適切な単語B', '不適切な単語C'];

    // 応答のチェック
    for (var i = 0; i < responseInappropriateWords.length; i++) {
      if (answer.includes(responseInappropriateWords[i])) {
        return "【システム通知】申し訳ありませんが、その内容には回答できません。";
      }
    }

    // ChatGPTの応答に注意文を追加
    answer += "\n\n＊ChatGPTは間違いを犯す可能性があります。詳しいことが知りたい場合は先生に確認して下さい。";

    // スプレッドシートに記録
    var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('履歴');
    sheet.appendRow([Utilities.formatDate(now, "JST", "yyyy-MM-dd HH:mm:ss"), email, talk_str, originalAnswer]);

    return answer;
  } else {
    return "no result";
  }
}

// 5分ごとにリセットを行う関数（トリガーに設定）
function resetAccesslog() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('アクセス制限検出');
  var data = sheet.getDataRange().getValues();

  // 2行目から最後までの行を削除
  if (data.length > 1) {
    sheet.deleteRows(2, data.length - 1);
  }
}

// 毎日リセットを行う関数（トリガーに設定する）
function dailyReset() {
  var sheet = SpreadsheetApp.getActiveSpreadsheet().getSheetByName('メールアドレス記録（毎日リセット）');
  var data = sheet.getDataRange().getValues();

  // 2行目から最後までの行を削除
  if (data.length > 1) {
    sheet.deleteRows(2, data.length - 1);
  }
}
