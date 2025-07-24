<%@ taglib uri="http://java.sun.com/jsp/jstl/core" prefix="c" %>
<%@ taglib uri="http://java.sun.com/jsp/jstl/fmt" prefix="fmt" %>
<%@ page language="java" contentType="text/html; charset=UTF-8"
    pageEncoding="UTF-8"%>
<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<title>농장 추가 페이지</title>
</head>
<body>
    <h1>농장 추가</h1>
		<form action="${pageContext.request.contextPath}/insertFarm" method="post" enctype="multipart/form-data">
        <ul>
            <li>
                <span>농장 이름</span>
                <input type="text" name="farmName" placeholder="농장 이름을 입력해주세요" class="ipt_tt" maxlength="100" required>
            </li>
            <li>
                <span>농장 주소</span>
                <input type="text" name="farmAddr" placeholder="농장 주소를 입력해주세요" class="ipt_tt" maxlength="200" required>
            </li>
            <li>
                <span>농장 전화번호</span>
                <input type="text" name="farmPhone" placeholder="농장 전화번호를 입력해주세요" class="ipt_tt" maxlength="20" required>
            </li>
            <li>
			    <span>농작물 종류</span>
			    <input type="text" name="farmCrops" placeholder="농작물을 입력해주세요" class="ipt_tt" maxlength="100" required>
			</li>

            <li>
                <span>농장 면적</span>
                <input type="text" name="farmArea" placeholder="농장 면적을 입력해주세요" class="ipt_tt" maxlength="50">
            </li>
            <li>
                <span>작성 날짜</span>
                <input type="date" name="createdAt" class="ipt_tt" required>
            </li>
            <li>
                <span>농장 이미지</span>
                <input type="file" name="farmImg" accept="image/*">
            </li>
            <li>
                <span>농장주 휴대폰번호</span>
                <input type="text" name="userPhone" placeholder="농장주 전화번호를 입력해주세요" class="ipt_tt" maxlength="20" required>
            </li>
            <li>
                <button type="submit">등록하기</button>
            </li>
        </ul>
    </form> 
</body>
</html>
